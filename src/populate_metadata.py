#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Populate bulk metadata tables from delimited text files.
"""

#
#  Copyright (C) 2011-2014 University of Dundee. All rights reserved.
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#


import logging
import gzip
import sys
import csv
import re
import json
from getpass import getpass
from getopt import getopt, GetoptError
from itertools import izip

import omero.clients
from omero import CmdError
from omero.rtypes import rlist, rstring, unwrap
from omero.model import DatasetAnnotationLinkI, DatasetI, FileAnnotationI
from omero.model import OriginalFileI, PlateI, PlateAnnotationLinkI, ScreenI
from omero.model import PlateAcquisitionI, WellI, WellSampleI, ImageI
from omero.model import ProjectAnnotationLinkI, ProjectI
from omero.model import ScreenAnnotationLinkI
from omero.model import MapAnnotationI, NamedValue
from omero.grid import ImageColumn, LongColumn, PlateColumn, RoiColumn
from omero.grid import StringColumn, WellColumn, DoubleColumn, BoolColumn
from omero.grid import DatasetColumn
from omero.util.metadata_mapannotations import (
    CanonicalMapAnnotation, MapAnnotationPrimaryKeyException,
    MapAnnotationManager)
from omero.util.metadata_utils import (
    KeyValueListPassThrough, KeyValueGroupList, NSBULKANNOTATIONSCONFIG)
from omero.util import pydict_text_io
from omero import client

from omero.util.populate_roi import ThreadPool

from omero_data_retriever import OmeroDataRetriever
from value_resolvers import *
from header_resolvers import *

log = logging.getLogger("omero.util.populate_metadata")


def usage(error):
    """Prints usage so that we don't have to. :)"""
    cmd = sys.argv[0]
    print """%s
Usage: %s [options] <target_object> <file>
Runs metadata population code for a given object.

Options:
  -s            OMERO hostname to use [defaults to "localhost"]
  -p            OMERO port to use [defaults to 4064]
  -u            OMERO username to use
  -w            OMERO password
  -k            OMERO session key to use
  --columns     Column configuration, Specify as comma separated list.
                Supported types: plate, well, image, roi,
                                 d (double), l (long), s (string), b (boolean)
                Supported Boolean True Values: "yes", "true", "t", "1".
  -i            Dump measurement information and exit (no population)
  -d            Print debug statements
  -c            Use an alternative context (for expert users only)

Examples:
  %s -s localhost -p 14064 -u bob --columns l,image,d,l Plate:6 metadata.csv

Report bugs to ome-devel@lists.openmicroscopy.org.uk""" % (error, cmd, cmd)
    sys.exit(2)


# Global thread pool for use by workers
thread_pool = None

# Special column names we may add depending on the data type
BOOLEAN_TRUE = ["yes", "true", "t", "1"]

PLATE_NAME_COLUMN = 'Plate Name'
WELL_NAME_COLUMN = 'Well Name'
DATASET_NAME_COLUMN = 'Dataset Name'
IMAGE_NAME_COLUMN = 'Image Name'

ADDED_COLUMN_NAMES = [PLATE_NAME_COLUMN,
                      WELL_NAME_COLUMN,
                      DATASET_NAME_COLUMN,
                      IMAGE_NAME_COLUMN,
                      'image']

COLUMN_TYPES = {
    'plate': PlateColumn, 'well': WellColumn, 'image': ImageColumn,
    'dataset': DatasetColumn, 'roi': RoiColumn,
    'd': DoubleColumn, 'l': LongColumn, 's': StringColumn, 'b': BoolColumn
}

DEFAULT_TABLE_NAME = 'bulk_annotations'
MAX_COLUMN_COUNT = 512


class Skip(object):
    """Instance to denote a row skip request."""
    pass


class MetadataError(Exception):
    """
    Raised by the metadata parsing context when an error condition
    is reached.
    """
    pass


class ParsingUtilFactory(object):

    def get_filter_for_plate(self, column_index, target_name):
        return lambda row: True if row[column_index] == target_name else False

    def get_generic_filter(self):
        return lambda row: True

    def __init__(self, target_object, data_retriever):
        self.target_object = target_object
        self.target_class = target_object.__class__
        if ScreenI is self.target_class:
            self.header_resolver = ScreenHeaderResolver()
            self.value_resolver = ScreenValueResolver(data_retriever, target_object)
        elif PlateI is self.target_class:
            self.header_resolver = PlateHeaderResolver()
            self.value_resolver = PlateValueResolver(data_retriever, target_object)
        elif DatasetI is self.target_class:
            self.header_resolver = DatasetHeaderResolver()
            self.value_resolver = DatasetValueResolver(data_retriever, target_object)
        elif ProjectI is self.target_class:
            self.header_resolver = ProjectHeaderResolver()
            self.value_resolver = ProjectValueResolver(data_retriever, target_object)
        else:
            raise MetadataError(
                'Unsupported target object class: %s' % self.target_class)

    def get_value_resolver(self):
        return self.value_resolver

    def get_header_resolver(self):
        return self.header_resolver

    #Filter Functions
    def get_filter_function(self, column_index=-1):
        if PlateI is self.target_class and column_index != -1:
            return self.get_filter_for_plate(
                column_index, unwrap(self.target_object.getName()))
        else:
            return self.get_generic_filter()

    def get_filter_for_plate(self, column_index, target_name):
        return lambda row : True if row[column_index] == target_name else False

    def get_generic_filter(self):
        return lambda row : True



class ParsingContext(object):
    """Generic parsing context for CSV files."""

    def __init__(self, client, target_object, parsing_util_factory, file=None,
                 fileid=None, cfg=None, cfgid=None, attach=False,
                 column_types=None, options=None, batch_size=1000,
                 loops=10, ms=500):
        '''
        This lines should be handled outside of the constructor:

        if not file:
            raise MetadataError('file required for %s' % type(self))
        if fileid and not file:
            raise MetadataError('fileid not supported for %s' % type(self))
        if cfg:
            raise MetadataError('cfg not supported for %s' % type(self))
        if cfgid:
            raise MetadataError('cfgid not supported for %s' % type(self))
        '''

        self.client = client
        self.target_object = target_object
        self.file = file
        self.column_types = column_types
        self.parsing_util_factory = parsing_util_factory
        self.value_resolver = self.parsing_util_factory.get_value_resolver()
        self.header_resolver = self.parsing_util_factory.get_header_resolver()

    def create_annotation_link(self):
        self.target_class = self.target_object.__class__
        if ScreenI is self.target_class:
            return ScreenAnnotationLinkI()
        if PlateI is self.target_class:
            return PlateAnnotationLinkI()
        if DatasetI is self.target_class:
            return DatasetAnnotationLinkI()
        if ProjectI is self.target_class:
            return ProjectAnnotationLinkI()
        raise MetadataError(
            'Unsupported target object class: %s' % self.target_class)

    def get_column_widths(self):
        widths = list()
        for column in self.columns:
            try:
                widths.append(column.size)
            except AttributeError:
                widths.append(None)
        return widths

    def preprocess_from_handle(self, data):
        reader = csv.reader(data, delimiter=',')
        first_row = reader.next()
        header_row = first_row
        first_row_is_types = HeaderResolver.is_row_column_types(first_row)
        header_index = 0
        if first_row_is_types:
            header_index = 1
            header_row = reader.next()
        log.debug('Header: %r' % header_row)
        for h in first_row:
            if not h:
                raise Exception('Empty column header in CSV: %s'
                                % first_row[header_index])
        if self.column_types is None and first_row_is_types:
            self.column_types = HeaderResolver.get_column_types(first_row)
        log.debug('Column types: %r' % self.column_types)
        self.columns = self.header_resolver.create_columns(header_row,
            self.column_types)
        log.debug('Columns: %r' % self.columns)
        if len(self.columns) > MAX_COLUMN_COUNT:
            log.warn("Column count exceeds max column count")

        self.preprocess_data(reader)

    def parse_from_handle_stream(self, data):
        reader = csv.reader(data, delimiter=',')
        first_row = reader.next()
        header_row = first_row
        first_row_is_types = HeaderResolver.is_row_column_types(first_row)
        if first_row_is_types:
            header_row = reader.next()

        plate_header_index = -1
        for i, name in enumerate(header_row):
            if name.lower() == 'plate':
                plate_header_index = i
                break

        self.filter_function = self.parsing_util_factory.get_filter_function(
            plate_header_index)

        table = self.create_table()
        self.populate_from_reader(reader, self.filter_function, table, 1000)
        self.create_file_annotation(table)

        self.post_process()
        log.debug('Column widths: %r' % self.get_column_widths())
        log.debug('Columns: %r' % [
            (o.name, len(o.values)) for o in self.columns])

    def create_table(self):
        sf = self.client.getSession()
        group = str(self.value_resolver.target_group)
        sr = sf.sharedResources()
        table = sr.newTable(1, DEFAULT_TABLE_NAME, {'omero.group': group})
        if table is None:
            raise MetadataError(
                "Unable to create table: %s" % DEFAULT_TABLE_NAME)
        original_file = table.getOriginalFile()
        log.info('Created new table OriginalFile:%d' % original_file.id.val)
        table.initialize(self.columns)
        return table

    def parse(self):
        if self.file.endswith(".gz"):
            data_for_preprocessing = gzip.open(self.file, "rb")
            data = gzip.open(self.file, "rb")
        else:
            data_for_preprocessing = open(self.file, 'U')
            data = open(self.file, 'U')

        try:
            self.preprocess_from_handle(data_for_preprocessing)
            return self.parse_from_handle_stream(data)
        finally:
            data.close()

    def preprocess_data(self, reader):
        for i, row in enumerate(reader):
            row = [(self.columns[i], value) for i, value in enumerate(row)]
            for column, original_value in row:
                log.debug('Original value %s, %s',
                          original_value, column.name)
                value = self.value_resolver.resolve(
                    column, original_value, row)
                if value.__class__ is Skip:
                    break
                try:
                    log.debug("Value's class: %s" % value.__class__)
                    if value.__class__ is str:
                        column.size = max(column.size, len(value))
                    # The following are needed for
                    # getting post process column sizes
                    if column.__class__ is WellColumn:
                        column.values.append(value)
                    elif column.__class__ is ImageColumn:
                        column.values.append(value)
                    elif column.name.lower() == "plate":
                        column.values.append(value)
                except TypeError:
                    log.error('Original value "%s" now "%s" of bad type!' % (
                        original_value, value))
                    raise
            self.post_process()
            for column in self.columns:
                column.values = []

    def populate_row(self, row):
        values = list()
        row = [(self.columns[i], value) for i, value in enumerate(row)]
        for column, original_value in row:
            log.debug('Original value %s, %s',
                      original_value, column.name)
            value = self.value_resolver.resolve(
                column, original_value, row)
            if value.__class__ is Skip:
                break
            values.append(value)
        if value.__class__ is not Skip:
            values.reverse()
            for column in self.columns:
                if not values:
                    if isinstance(column, ImageColumn) or \
                       column.name in (PLATE_NAME_COLUMN,
                                       WELL_NAME_COLUMN,
                                       IMAGE_NAME_COLUMN):
                        # Then assume that the values will be calculated
                        # later based on another column.
                        continue
                    else:
                        msg = 'Column %s has no values.' % column.name
                        log.error(msg)
                        raise IndexError(msg)
                else:
                    column.values.append(values.pop())

    def populate_from_reader(self,
                             reader,
                             filter_function,
                             table,
                             batch_size=1000):
        row_count = 0
        for (r, row) in enumerate(reader):
            log.debug('Row %d', r)
            if filter_function(row):
                self.populate_row(row)
                row_count = row_count + 1
                if row_count >= batch_size:
                    self.post_process()
                    table.addData(self.columns)
                    for column in self.columns:
                        column.values = []
                    row_count = 0
        if row_count != 0:
            log.debug("DATA TO ADD")
            log.debug(self.columns)
            self.post_process()
            table.addData(self.columns)
        table.close()

    def create_file_annotation(self, table):
        sf = self.client.getSession()
        group = str(self.value_resolver.target_group)
        update_service = sf.getUpdateService()

        original_file = table.getOriginalFile()
        file_annotation = FileAnnotationI()
        file_annotation.ns = rstring(
            'openmicroscopy.org/omero/bulk_annotations')
        file_annotation.description = rstring(DEFAULT_TABLE_NAME)
        file_annotation.file = OriginalFileI(original_file.id.val, False)
        link = self.create_annotation_link()
        link.parent = self.target_object
        link.child = file_annotation
        update_service.saveObject(link, {'omero.group': group})

    def populate(self, rows):
        nrows = len(rows)
        for (r, row) in enumerate(rows):
            log.debug('Row %d/%d', r + 1, nrows)
            self.populate_row(row)

    def post_process(self):
        target_class = self.target_object.__class__
        columns_by_name = dict()
        well_column = None
        well_name_column = None
        plate_name_column = None
        image_column = None
        image_name_column = None
        resolve_image_names = False
        resolve_image_ids = False
        for column in self.columns:
            columns_by_name[column.name.lower()] = column
            if column.__class__ is PlateColumn:
                log.warn("PlateColumn is unimplemented")
            elif column.__class__ is WellColumn:
                well_column = column
            elif column.name == WELL_NAME_COLUMN:
                well_name_column = column
            elif column.name == PLATE_NAME_COLUMN:
                plate_name_column = column
            elif column.name == IMAGE_NAME_COLUMN:
                image_name_column = column
                log.debug("Image name column len:%d" % len(column.values))
                if len(column.values) > 0:
                    resolve_image_ids = True
                    log.debug("Resolving Image Ids")
            elif column.__class__ is ImageColumn:
                image_column = column
                log.debug("Image column len:%d" % len(column.values))
                if len(column.values) > 0:
                    resolve_image_names = True
                    log.debug("Resolving Image Ids")

        log.debug("Column by name:%r" % columns_by_name)
        if well_name_column is None and plate_name_column is None \
                and image_name_column is None:
            log.info('Nothing to do during post processing.')
            return

        sz = max([len(x.values) for x in self.columns])
        for i in range(0, sz):
            if well_name_column is not None:

                v = ''
                try:
                    well_id = well_column.values[i]
                    plate = None
                    if "plate" in columns_by_name:  # FIXME
                        plate = columns_by_name["plate"].values[i]
                    v = self.value_resolver.get_well_name(well_id, plate)
                except KeyError:
                    log.warn(
                        'Skipping table row %d! Missing well row or column '
                        'for well name population!' % i, exc_info=True
                    )
                well_name_column.size = max(well_name_column.size, len(v))
                well_name_column.values.append(v)
            else:
                log.info('Missing well name column, skipping.')

            if image_name_column is not None and (
                    DatasetI is target_class or
                    ProjectI is target_class) and \
                    resolve_image_names and not resolve_image_ids:
                iname = ""
                try:
                    log.debug(image_name_column)
                    iid = image_column.values[i]
                    did = self.target_object.id.val
                    if "dataset name" in columns_by_name:
                        dname = columns_by_name["dataset name"].values[i]
                        did = self.value_resolver.datasets_by_name[
                            dname].id.val
                    elif "dataset" in columns_by_name:
                        did = int(columns_by_name["dataset"].values[i])
                    log.debug("Using Dataset:%d" % did)
                    iname = self.value_resolver.get_image_name_by_id(
                        iid, did)
                except KeyError:
                    log.warn(
                        "%d not found in image ids" % iid)
                assert i == len(image_name_column.values)
                image_name_column.values.append(iname)
                image_name_column.size = max(
                    image_name_column.size, len(iname))
            elif image_name_column is not None and (
                    DatasetI is target_class or
                    ProjectI is target_class) and \
                    resolve_image_ids and not resolve_image_names:
                iid = -1
                try:
                    log.debug(image_column)
                    iname = image_name_column.values[i]
                    did = self.target_object.id.val
                    if "dataset name" in columns_by_name:
                        dname = columns_by_name["dataset name"].values[i]
                        did = self.value_resolver.datasets_by_name[
                            dname].id.val
                    elif "dataset" in columns_by_name:
                        did = int(columns_by_name["dataset"].values[i])
                    log.debug("Using Dataset:%d" % did)
                    iid = self.value_resolver.get_image_id_by_name(
                        iname, did)
                except KeyError:
                    log.warn(
                        "%d not found in image ids" % iid)
                assert i == len(image_column.values)
                image_column.values.append(iid)
            elif image_name_column is not None and (
                    ScreenI is target_class or
                    PlateI is target_class):
                iid = image_column.values[i]
                log.info("Checking image %s", iid)
                pid = None
                if 'plate' in columns_by_name:
                    pid = columns_by_name['plate'].values[i]
                iname = self.value_resolver.get_image_name_by_id(iid, pid)
                image_name_column.values.append(iname)
                image_name_column.size = max(
                    image_name_column.size, len(iname)
                )
            else:
                log.info('Missing image name column, skipping.')

            if plate_name_column is not None:
                plate = columns_by_name['plate'].values[i]   # FIXME
                v = self.value_resolver.get_plate_name_by_id(plate)
                plate_name_column.size = max(plate_name_column.size, len(v))
                plate_name_column.values.append(v)
            else:
                log.info('Missing plate name column, skipping.')


class _QueryContext(object):
    """
    Helper class container query methods
    """
    def __init__(self, client):
        self.client = client

    def _batch(self, i, sz=1000):
        """
        Generate batches of size sz (by default 1000) from the input
        iterable `i`.
        """
        i = list(i)  # Copying list to handle sets and modifications
        for batch in (i[pos:pos + sz] for pos in xrange(0, len(i), sz)):
            yield batch

    def _grouped_batch(self, groups, sz=1000):
        """
        In some cases groups of objects must be kept together.
        This method attempts to return up to sz objects without breaking
        up groups.
        If a single group is greater than sz the whole group is returned.
        Callers are responsible for checking the size of the returned batch.

        :param groups: an iterable of lists/tuples of objects
        :return: a list of the next batch of objects, if a single group has
                 more than sz elements it will be returned as a single
                 over-sized batch
        """
        batch = []
        for group in groups:
            if (len(batch) == 0) or (len(batch) + len(group) <= sz):
                batch.extend(group)
            else:
                toyield = batch
                batch = group
                yield toyield
        if batch:
            yield batch

    def projection(self, q, ids, nss=None, batch_size=None):
        """
        Run a projection query designed to return scalars only
        :param q: The query to be projected, should contain either `:ids`
               or `:id` as a parameter
        :param: ids: Either a list of IDs to be passed as `:ids` parameter or
                a single scalar id to be passed as `:id` parameter in query
        :nss: Optional, Either a list of namespaces to be passed as `:nss`
                parameter or a single string to be passed as `:ns` parameter
                in query
        :batch_size: Optional batch_size (default: all) defining the number
                of IDs that will be queried at once. Methods that expect to
                have more than several thousand input IDs should consider an
                appropriate batch size. By default, however, no batch size is
                applied since this could change the interpretation of the
                query string (e.g. use of `distinct`).
        """
        qs = self.client.getSession().getQueryService()
        params = omero.sys.ParametersI()

        try:
            nids = len(ids)
            single_id = None
        except TypeError:
            nids = 1
            single_id = ids

        if isinstance(nss, basestring):
            params.addString("ns", nss)
        elif nss:
            params.map['nss'] = rlist(rstring(s) for s in nss)

        log.debug("Query: %s len(IDs): %d namespace(s): %s", q, nids, nss)

        if single_id is not None:
            params.addId(single_id)
            rss = unwrap(qs.projection(q, params))
        elif batch_size is None:
            params.addIds(ids)
            rss = unwrap(qs.projection(q, params))
        else:
            rss = []
            for batch in self._batch(ids, sz=batch_size):
                params.addIds(batch)
                rss.extend(unwrap(qs.projection(q, params)))

        return [r for rs in rss for r in rs]


def get_config(session, cfg=None, cfgid=None):
    if cfgid:
        cfgdict = pydict_text_io.load(
            'OriginalFile:%d' % cfgid, session=session)
    elif cfg:
        cfgdict = pydict_text_io.load(cfg)
    else:
        raise Exception("Configuration file required")

    default_cfg = cfgdict.get("defaults")
    column_cfgs = cfgdict.get("columns")
    advanced_cfgs = cfgdict.get("advanced", {})
    if not default_cfg and not column_cfgs:
        raise Exception(
            "Configuration defaults and columns were both empty")
    return default_cfg, column_cfgs, advanced_cfgs


class BulkToMapAnnotationContext(_QueryContext):
    """
    Processor for creating MapAnnotations from BulkAnnotations.
    """

    def __init__(self, client, target_object, file=None, fileid=None,
                 cfg=None, cfgid=None, attach=False, options=None,
                 batch_size=1000, loops=10, ms=10, dry_run=False):
        """
        :param client: OMERO client object
        :param target_object: The object to be annotated
        :param file: Not supported
        :param fileid: The OriginalFile ID of the bulk-annotations table,
               default is to use the a bulk-annotation attached to
               target_object
        :param cfg: Path to a configuration file, ignored if cfgid given
        :param cfgid: OriginalFile ID of configuration file, either cfgid or
               cfg must be given
        """
        super(BulkToMapAnnotationContext, self).__init__(client)

        if file and not fileid:
            raise MetadataError('file not supported for %s' % type(self))

        # Reload object to get .details
        self.target_object = self.get_target(target_object)
        if fileid:
            self.ofileid = fileid
        else:
            self.ofileid = self.get_bulk_annotation_file()
        if not self.ofileid:
            raise MetadataError("Unable to find bulk-annotations file")

        self.default_cfg, self.column_cfgs, self.advanced_cfgs = \
            get_config(self.client.getSession(), cfg=cfg, cfgid=cfgid)

        self.pkmap = {}
        self.mapannotations = MapAnnotationManager()
        self._init_namespace_primarykeys()

        self.options = {}
        if options:
            self.options = options
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = 1000

    def _init_namespace_primarykeys(self):
        try:
            pkcfg = self.advanced_cfgs['primary_group_keys']
        except (TypeError, KeyError):
            return None

        for pk in pkcfg:
            try:
                gns = pk['namespace']
                keys = pk['keys']
            except KeyError:
                raise Exception('Invalid primary_group_keys: %s' % pk)
            if keys:
                if not isinstance(keys, list):
                    raise Exception('keys must be a list')
                if gns in self.pkmap:
                    raise Exception('Duplicate namespace in keys: %s' % gns)

                self.pkmap[gns] = keys
                self.mapannotations.add_from_namespace_query(
                    self.client.getSession(), gns, keys)
                log.debug('Loaded ns:%s primary-keys:%s', gns, keys)

    def _get_ns_primary_keys(self, ns):
        return self.pkmap.get(ns, None)

    def _get_selected_namespaces(self):
        try:
            nss = self.options['ns']
            if isinstance(nss, list):
                return nss
            return [nss]
        except KeyError:
            return None

    def get_target(self, target_object):
        qs = self.client.getSession().getQueryService()
        return qs.find(target_object.ice_staticId().split('::')[-1],
                       target_object.id.val)

    def get_bulk_annotation_file(self):
        otype = self.target_object.ice_staticId().split('::')[-1]
        q = """SELECT child.file.id FROM %sAnnotationLink link
               WHERE parent.id=:id AND child.ns=:ns ORDER by id""" % otype
        r = self.projection(q, unwrap(self.target_object.getId()),
                            omero.constants.namespaces.NSBULKANNOTATIONS)
        if r:
            return r[-1]

    def _create_cmap_annotation(self, targets, rowkvs, ns):
        pks = self._get_ns_primary_keys(ns)

        ma = MapAnnotationI()
        ma.setNs(rstring(ns))
        mv = []
        for k, vs in rowkvs:
            if not isinstance(vs, (tuple, list)):
                vs = [vs]
            mv.extend(NamedValue(k, str(v)) for v in vs)

        if not mv:
            log.debug('Empty MapValue, ignoring: %s', rowkvs)
            return

        ma.setMapValue(mv)

        log.debug('Creating CanonicalMapAnnotation ns:%s pks:%s kvs:%s',
                  ns, pks, rowkvs)
        cma = CanonicalMapAnnotation(ma, primary_keys=pks)
        for (otype, oid) in targets:
            cma.add_parent(otype, oid)
        return cma

    def _create_map_annotation_links(self, cma):
        """
        Converts a `CanonicalMapAnnotation` object into OMERO `MapAnnotations`
        and `AnnotationLinks`. `AnnotationLinks` will have the parent and
        child set, where the child is a single `MapAnnotation` object shared
        amongst all links.

        If the number of `AnnotationLinks` is small callers may choose to save
        the links and `MapAnnotation` using `UpdateService` directly, ignoring
        the second element of the tuple.

        If the number of `AnnotationLinks` is large the caller should first
        save the `MapAnnotation`, call `setChild` on all `AnnotationLinks` to
        reference the saved `MapAnnotation`, and finally save all
        `AnnotationLinks`.

        This complexity is unfortunately required to avoid going over the
        Ice message size.

        :return: A tuple of (`AnnotationLinks`, `MapAnnotation`)
        """
        links = []
        ma = cma.get_mapann()
        for (otype, oid) in cma.get_parents():
            link = getattr(omero.model, '%sAnnotationLinkI' % otype)()
            link.setParent(getattr(omero.model, '%sI' % otype)(oid, False))
            link.setChild(ma)
            links.append(link)
        return links, ma

    def _save_annotation_links(self, links):
        """
        Save `AnnotationLinks` including the child annotation in one go.
        See `_create_map_annotation_links`
        """
        sf = self.client.getSession()
        group = str(self.target_object.details.group.id)
        update_service = sf.getUpdateService()
        arr = update_service.saveAndReturnArray(links, {'omero.group': group})
        return arr

    def _save_annotation_and_links(self, links, ann, batch_size):
        """
        Save a single `Annotation`, followed by the `AnnotationLinks` to that
        Annotation and return the number of links saved.

        All `AnnotationLinks` must have `ann` as their child.
        Links will be saved in batches of `batch_size`.

        See `_create_map_annotation_links`
        """
        sf = self.client.getSession()
        group = str(self.target_object.details.group.id)
        update_service = sf.getUpdateService()

        annobj = update_service.saveAndReturnObject(ann)
        annobj.unload()

        sz = 0
        for batch in self._batch(links, sz=batch_size):
            for link in batch:
                link.setChild(annobj)
            update_service.saveArray(
                batch, {'omero.group': group})
            sz += len(batch)
        return sz

    def parse(self):
        tableid = self.ofileid
        sr = self.client.getSession().sharedResources()
        log.debug('Loading table OriginalFile:%d', self.ofileid)
        table = sr.openTable(omero.model.OriginalFileI(tableid, False))
        assert table

        try:
            self.populate(table)
        finally:
            table.close()
        self.write_to_omero()

    def _get_additional_targets(self, target):
        iids = []
        try:
            if self.advanced_cfgs['well_to_images'] and target[0] == 'Well':
                q = 'SELECT image.id FROM WellSample WHERE well.id=:id'
                iids = self.projection(q, target[1])
        except (KeyError, TypeError):
            pass
        return [('Image', i) for i in iids]

    def populate(self, table):
        def idcolumn_to_omeroclass(col):
            clsname = re.search('::(\w+)Column$', col.ice_staticId()).group(1)
            return clsname

        try:
            ignore_missing_primary_key = self.advanced_cfgs[
                'ignore_missing_primary_key']
        except (KeyError, TypeError):
            ignore_missing_primary_key = False

        nrows = table.getNumberOfRows()
        data = table.readCoordinates(range(nrows))

        # Don't create annotations on higher-level objects
        # idcoltypes = set(HeaderResolver.screen_keys.values())
        idcoltypes = set((ImageColumn, WellColumn))
        idcols = []
        for n in xrange(len(data.columns)):
            col = data.columns[n]
            if col.__class__ in idcoltypes:
                omeroclass = idcolumn_to_omeroclass(col)
                idcols.append((omeroclass, n))

        headers = [c.name for c in data.columns]
        if self.default_cfg or self.column_cfgs:
            kvgl = KeyValueGroupList(
                headers, self.default_cfg, self.column_cfgs)
            trs = kvgl.get_transformers()
        else:
            trs = [KeyValueListPassThrough(headers)]

        selected_nss = self._get_selected_namespaces()
        for row in izip(*(c.values for c in data.columns)):
            targets = []
            for omerotype, n in idcols:
                if row[n] > 0:
                    # Be aware this has implications for client UIs, since
                    # Wells and Images may be treated as one when it comes
                    # to annotations
                    obj = (omerotype, row[n])
                    targets.append(obj)
                    targets.extend(self._get_additional_targets(obj))
                else:
                    log.warn("Invalid Id:%d found in row %s", row[n], row)
            if targets:
                for tr in trs:
                    rowkvs = tr.transform(row)
                    ns = tr.name
                    if not ns:
                        ns = omero.constants.namespaces.NSBULKANNOTATIONS
                    if (selected_nss is not None) and (ns not in selected_nss):
                        log.debug('Skipping namespace: %s', ns)
                        continue
                    try:
                        cma = self._create_cmap_annotation(targets, rowkvs, ns)
                        if cma:
                            self.mapannotations.add(cma)
                            log.debug('Added MapAnnotation: %s', cma)
                        else:
                            log.debug(
                                'Empty MapAnnotation: %s', rowkvs)
                    except MapAnnotationPrimaryKeyException as e:
                        c = ''
                        if ignore_missing_primary_key:
                            c = ' (Continuing)'
                        log.error(
                            'Missing primary keys%s: %s %s ', c, e, rowkvs)
                        if not ignore_missing_primary_key:
                            raise

    def _write_log(self, text):
        log.debug("BulkToMapAnnotation:write_to_omero - %s" % text)

    def write_to_omero(self):
        i = 0
        cur = 0
        links = []

        # This may be many-links-to-one-new-mapann so everything must
        # be kept together to avoid duplication of the mapann
        self._write_log("Start")
        cmas = self.mapannotations.get_map_annotations()
        self._write_log("found %s annotations" % len(cmas))
        for cma in cmas:
            batch, ma = self._create_map_annotation_links(cma)
            self._write_log("found batch of size %s" % len(batch))
            if len(batch) < self.batch_size:
                links.append(batch)
                cur += len(batch)
                if cur > 10 * self.batch_size:
                    self._write_log("running batches. accumulated: %s" % cur)
                    i += self._write_links(links, self.batch_size, i)
                    links = []
                    cur = 0
            else:
                self._write_log("running grouped_batch")
                print("_save_annotation_and_links with batch size %d" %
                      (self.batch_size))
                sz = self._save_annotation_and_links(batch, ma,
                                                     self.batch_size)
                i += sz
                log.info('Created/linked %d MapAnnotations (total %s)',
                         sz, i)
        # Handle any remaining writes
        i += self._write_links(links, self.batch_size, i)

    def _write_links(self, links, batch_size, i):
        count = 0
        for batch in self._grouped_batch(links, sz=batch_size):
            self._write_log("batch size: %s" % len(batch))
            arr = self._save_annotation_links(batch)
            count += len(arr)
            log.info('Created/linked %d MapAnnotations (total %s)',
                     len(arr), i+count)
        return count


class DeleteMapAnnotationContext(_QueryContext):
    """
    Processor for deleting MapAnnotations in the BulkAnnotations namespace
    on these types: Image WellSample Well PlateAcquisition Plate Screen
    """

    def __init__(self, client, target_object, file=None, fileid=None,
                 cfg=None, cfgid=None, attach=False, options=None,
                 batch_size=1000, loops=10, ms=500, dry_run=False):

        """
        :param client: OMERO client object
        :param target_object: The object to be processed
        :param file, fileid: Ignored
        :param cfg, cfgid: Configuration file
        :param attach: Delete all attached config files (recursive,
               default False)
        """
        super(DeleteMapAnnotationContext, self).__init__(client)
        self.target_object = target_object
        self.attach = attach

        if cfg or cfgid:
            self.default_cfg, self.column_cfgs, self.advanced_cfgs = \
                get_config(self.client.getSession(), cfg=cfg, cfgid=cfgid)
        else:
            self.default_cfg = None
            self.column_cfgs = None
            self.advanced_cfgs = None

        self.options = {}
        if options:
            self.options = options
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = 1000
        self.loops = loops
        self.ms = ms

    def parse(self):
        self.populate()
        self.write_to_omero()

    def _get_annotations_for_deletion(
            self, objtype, objids, anntype, nss, getlink=False):
        r = []
        if getlink:
            fetch = ''
        else:
            fetch = 'child.'
        if objids:
            q = ("SELECT %sid FROM %sAnnotationLink WHERE "
                 "child.class=%s AND parent.id in (:ids) "
                 "AND child.ns in (:nss)")
            r = self.projection(q % (fetch, objtype, anntype), objids, nss,
                                batch_size=10000)
            log.debug("%s: %d %s(s)", objtype, len(set(r)), anntype)
        return r

    def _get_configured_namespaces(self):
        try:
            nss = self.options['ns']
            if isinstance(nss, list):
                return nss
            return [nss]
        except KeyError:
            pass

        nss = set([
            omero.constants.namespaces.NSBULKANNOTATIONS,
            NSBULKANNOTATIONSCONFIG,
        ])
        if self.column_cfgs:
            for c in self.column_cfgs:
                try:
                    ns = c['group']['namespace']
                    nss.add(ns)
                except KeyError:
                    continue
        return list(nss)

    def populate(self):
        # Hierarchy: Screen, Plate, {PlateAcquistion, Well}, WellSample, Image
        parentids = {
            "Screen": None,
            "Plate":  None,
            "PlateAcquisition": None,
            "Well": None,
            "WellSample": None,
            "Image": None,
            "Dataset": None,
            "Project": None,
        }

        target = self.target_object
        ids = [unwrap(target.getId())]

        if isinstance(target, ScreenI):
            q = ("SELECT child.id FROM ScreenPlateLink "
                 "WHERE parent.id in (:ids)")
            parentids["Screen"] = ids
        if parentids["Screen"]:
            parentids["Plate"] = self.projection(q, parentids["Screen"])

        if isinstance(target, PlateI):
            parentids["Plate"] = ids
        if parentids["Plate"]:
            q = "SELECT id FROM PlateAcquisition WHERE plate.id IN (:ids)"
            parentids["PlateAcquisition"] = self.projection(
                q, parentids["Plate"])
            q = "SELECT id FROM Well WHERE plate.id IN (:ids)"
            parentids["Well"] = self.projection(q, parentids["Plate"])

        if isinstance(target, PlateAcquisitionI):
            parentids["PlateAcquisition"] = ids
        if parentids["PlateAcquisition"] and not isinstance(target, PlateI):
            # WellSamples are linked to PlateAcqs and Plates, so only get
            # if they haven't been obtained via a Plate
            # Also note that we do not get Wells if the parent is a
            # PlateAcquisition since this only refers to the fields in
            # the well
            q = "SELECT id FROM WellSample WHERE plateAcquisition.id IN (:ids)"
            parentids["WellSample"] = self.projection(
                q, parentids["PlateAcquisition"])

        if isinstance(target, WellI):
            parentids["Well"] = ids
        if parentids["Well"]:
            q = "SELECT id FROM WellSample WHERE well.id IN (:ids)"
            parentids["WellSample"] = self.projection(
                q, parentids["Well"], batch_size=10000)

        if isinstance(target, WellSampleI):
            parentids["WellSample"] = ids
        if parentids["WellSample"]:
            q = "SELECT image.id FROM WellSample WHERE id IN (:ids)"
            parentids["Image"] = self.projection(
                q, parentids["WellSample"], batch_size=10000)

        if isinstance(target, ProjectI):
            parentids["Project"] = ids
        if parentids["Project"]:
            q = ("SELECT ds.id FROM ProjectDatasetLink link "
                 "join link.parent prj "
                 "join link.child as ds WHERE prj.id IN (:ids)")
            parentids["Dataset"] = self.projection(q, parentids["Project"])

        if isinstance(target, DatasetI):
            parentids["Dataset"] = ids
        if parentids["Dataset"]:
            q = ("SELECT i.id FROM DatasetImageLink link "
                 "join link.parent ds "
                 "join link.child as i WHERE ds.id IN (:ids)")
            parentids["Image"] = self.projection(q, parentids["Dataset"])

        if isinstance(target, ImageI):
            parentids["Image"] = ids

        # TODO: This should really include:
        #    raise Exception("Unknown target: %s" % target.__class__.__name__)

        log.debug("Parent IDs: %s",
                  ["%s:%s" % (k, v is not None and len(v) or "NA")
                   for k, v in parentids.items()])

        self.mapannids = dict()
        self.fileannids = set()
        not_annotatable = ('WellSample',)

        # Currently deleting AnnotationLinks should automatically delete
        # orphaned MapAnnotations:
        # https://github.com/openmicroscopy/openmicroscopy/pull/4907
        # Note this may change in future:
        # https://trello.com/c/Gnoi9mTM/141-never-delete-orphaned-map-annotations
        nss = self._get_configured_namespaces()
        for objtype, objids in parentids.iteritems():
            if objtype in not_annotatable:
                continue
            r = self._get_annotations_for_deletion(
                objtype, objids, 'MapAnnotation', nss, getlink=True)
            if r:
                try:
                    self.mapannids[objtype].update(r)
                except KeyError:
                    self.mapannids[objtype] = set(r)

        log.info("Total MapAnnotationLinks in %s: %d",
                 nss, sum(len(v) for v in self.mapannids.values()))
        log.debug("MapAnnotationLinks in %s: %s", nss, self.mapannids)

        if self.attach and NSBULKANNOTATIONSCONFIG in nss:
            for objtype, objids in parentids.iteritems():
                if objtype in not_annotatable:
                    continue
                r = self._get_annotations_for_deletion(
                    objtype, objids, 'FileAnnotation',
                    [NSBULKANNOTATIONSCONFIG])
                self.fileannids.update(r)

            log.info("Total FileAnnotations in %s: %d",
                     [NSBULKANNOTATIONSCONFIG], len(set(self.fileannids)))
            log.debug("FileAnnotations in %s: %s",
                      [NSBULKANNOTATIONSCONFIG], self.fileannids)

    def write_to_omero(self):
        for objtype, maids in self.mapannids.iteritems():
            for batch in self._batch(maids, sz=self.batch_size):
                self._write_to_omero_batch(
                    {"%sAnnotationLink" % objtype: batch}, self.loops, self.ms)
        for batch in self._batch(self.fileannids, sz=self.batch_size):
            self._write_to_omero_batch({"FileAnnotation": batch},
                                       self.loops, self.ms)

    def _write_to_omero_batch(self, to_delete, loops=10, ms=500):
        del_cmd = omero.cmd.Delete2(targetObjects=to_delete)
        try:
            callback = self.client.submit(
                del_cmd, loops=loops, ms=ms, failontimeout=True)
        except CmdError, ce:
            log.error("Failed to delete: %s" % to_delete)
            raise Exception(ce.err)

        # At this point, we're sure that there's a response OR
        # an exception has been thrown (likely LockTimeout)
        rsp = callback.getResponse()
        try:
            deleted = rsp.deletedObjects
            for k, v in deleted.iteritems():
                log.info("Deleted: %s %d", k, len(v))
                log.debug("Deleted: %s %s", k, v)
        except AttributeError:
            log.error("Delete failed: %s", rsp)


def parse_target_object(target_object):
    type, id = target_object.split(':')
    if 'Dataset' == type:
        return DatasetI(long(id), False)
    if 'Project' == type:
        return ProjectI(long(id), False)
    if 'Plate' == type:
        return PlateI(long(id), False)
    if 'Screen' == type:
        return ScreenI(long(id), False)
    raise ValueError('Unsupported target object: %s' % target_object)


if __name__ == "__main__":
    try:
        options, args = getopt(sys.argv[1:], "s:p:u:w:k:c:id", ["columns="])
    except GetoptError, (msg, opt):
        usage(msg)

    try:
        target_object, file = args
        target_object = parse_target_object(target_object)
    except ValueError:
        usage('Target object and file must be a specified!')

    username = None
    password = None
    hostname = 'localhost'
    port = 4064  # SSL
    session_key = None
    logging_level = logging.INFO
    thread_count = 1
    column_types = None
    context_class = ParsingContext
    for option, argument in options:
        if option == "-u":
            username = argument
        if option == "-w":
            password = argument
        if option == "-s":
            hostname = argument
        if option == "-p":
            port = int(argument)
        if option == "-k":
            session_key = argument
        if option == "-d":
            logging_level = logging.DEBUG
        if option == "-t":
            thread_count = int(argument)
        if option == "--columns":
            column_types = parse_column_types(argument.split(','))
        if option == "-c":
            try:
                context_class = globals()[argument]
            except KeyError:
                usage("Invalid context class")
    if session_key is None and username is None:
        usage("Username must be specified!")
    if session_key is None and hostname is None:
        usage("Host name must be specified!")
    if session_key is None and password is None:
        password = getpass()

    logging.basicConfig(level=logging_level)
    client = client(hostname, port)
    client.setAgent("OMERO.populate_metadata")
    client.enableKeepAlive(60)
    try:
        if session_key is not None:
            client.joinSession(session_key)
            client.sf.detachOnDestroy()
        else:
            client.createSession(username, password)

        log.debug('Creating pool of %d threads' % thread_count)
        thread_pool = ThreadPool(thread_count)
        data_retriever = OmeroDataRetriever(client)
        parsing_util_factory = ParsingUtilFactory(target_object, data_retriever)
        ctx = context_class(
            client,
            target_object,
            file=file,
            column_types=column_types)

        ctx.parse()
    finally:
        pass
        client.closeSession()
