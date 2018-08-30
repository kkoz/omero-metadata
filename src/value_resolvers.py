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
import re
from collections import defaultdict

import omero.clients
from omero.rtypes import unwrap
from omero.grid import ImageColumn, LongColumn, PlateColumn, RoiColumn
from omero.grid import StringColumn, WellColumn, DoubleColumn, BoolColumn
from omero.grid import DatasetColumn


log = logging.getLogger("omero.util.value_resolvers")

class PlateData(object):
    """
    Largely "mock" object which is intended to simulate the data returned
    by querying a Plate but without the overhead of storing all the Ice
    fields.
    """

    def __init__(self, plate):
        self.id = plate.id
        self.name = plate.name
        self.wells = []
        for well in plate.copyWells():
            self.wells.append(WellData(well))


class WellData(object):
    """
    Largely "mock" object which is intended to simulate the data returned
    by querying a Well but without the overhead of storing all the Ice
    fields.
    """

    def __init__(self, well):
        self.id = well.id
        self.row = well.row
        self.column = well.column
        self.well_samples = []
        for well_sample in well.copyWellSamples():
            self.well_samples.append(WellSampleData(well_sample))

    def __eq__(self, other):
        if self.id.val != other.id.val \
            or self.row != other.row \
            or self.column != other.column:
            return False
        for i in range(0, len(self.well_samples)):
            if not(self.well_samples[i] == other.well_samples[i]):
                return False
        return True


class WellSampleData(object):
    """
    Largely "mock" object which is intended to simulate the data returned
    by querying a WellSample but without the overhead of storing all the Ice
    fields.
    """

    def __init__(self, well_sample):
        self.id = well_sample.id
        self.image = ImageData(well_sample.getImage())

    def __eq__(self, other):
        if self.id.val != other.id.val or not(self.image == other.image):
            return False
        return True


class ImageData(object):
    """
    Largely "mock" object which is intended to simulate the data returned
    by querying a Image but without the overhead of storing all the Ice
    fields.
    """

    def __init__(self, image):
        self.id = image.id
        self.name = image.name

    def __eq__(self, other):
        if self.id.val !=other.id.val or self.name.val != other.name.val:
            return False
        return True



class ValueResolver(object):
    """
    Value resolver for column types which is responsible for filling up
    non-metadata columns with their OMERO data model identifiers.
    """

    AS_ALPHA = [chr(v) for v in range(97, 122 + 1)]  # a-z
    # Support more than 26 rows
    for v in range(97, 122 + 1):
        AS_ALPHA.append('a' + chr(v))
    WELL_REGEX = re.compile(r'^([a-zA-Z]+)(\d+)$')

    def __init__(self, data_retriever, target_object):
        self.data_retriever = data_retriever
        self.target_object = target_object
        self.target_class = self.target_object.__class__
        self.target_group = self.data_retriever.get_target_group(target_object)

    def get_well_name(self, well_id, plate=None):
        well = self.get_well_by_id(well_id, plate)
        row = well.row.val
        col = well.column.val
        row = self.AS_ALPHA[row]
        return '%s%d' % (row, col + 1)

    def resolve(self, column, value, row):
        images_by_id = None
        column_class = column.__class__
        column_as_lower = column.name.lower()
        if ImageColumn is column_class:
            if len(self.images_by_id) == 1:
                images_by_id = self.images_by_id.values()[0]
            else:
                for column, column_value in row:
                    if column.__class__ is PlateColumn:
                        images_by_id = self.images_by_id[
                            self.plates_by_name[column_value].id.val
                        ]
                        log.debug(
                            "Got plate %i",
                            self.plates_by_name[column_value].id.val
                        )
                        break
                    elif column.name.lower() == "dataset name":
                        # DatasetColumn unimplemented at the momnet
                        # We can still access column names though
                        images_by_id = self.images_by_id[
                            self.datasets_by_name[column_value].id.val
                        ]
                        log.debug(
                            "Got dataset %i",
                            self.datasets_by_name[column_value].id.val
                        )
                        break
                    elif column.name.lower() == "dataset":
                        # DatasetColumn unimplemented at the momnet
                        # We can still access column names though
                        images_by_id = self.images_by_id[
                            self.datasets_by_id[
                                int(column_value)].id.val
                        ]
                        log.debug(
                            "Got dataset %i",
                            self.datasets_by_id[
                                int(column_value)].id.val
                        )
                        break
            if images_by_id is None:
                raise MetadataError(
                    'Unable to locate Parent column in Row: %r' % row
                )
            try:
                return images_by_id[long(value)].id.val
            except KeyError:
                log.debug('Image Id: %i not found!' % (value))
                return -1L
            return
        if WellColumn is column_class:
            return self.resolve_well(column, row, value)
        if PlateColumn is column_class:
            return self.resolve_plate(value)
        # Prepared to handle DatasetColumn
        if DatasetColumn is column_class:
            return self.resolve_dataset(column, row, value)
        if column_as_lower in ('row', 'column') \
           and column_class is LongColumn:
            try:
                # The value is not 0 offsetted
                return long(value) - 1
            except ValueError:
                return long(self.AS_ALPHA.index(value.lower()))
        if StringColumn is column_class:
            return value
        if LongColumn is column_class:
            return long(value)
        if DoubleColumn is column_class:
            return float(value)
        if BoolColumn is column_class:
            return value.lower() in BOOLEAN_TRUE
        raise MetadataError('Unsupported column class: %s' % column_class)


class SPWValueResolver(ValueResolver):

    def get_well_by_id(self, well_id, plate=None):
        raise Exception("to be implemented by subclasses")

    def get_image_name_by_id(self, iid, pid=None):
        if not pid and len(self.images_by_id):
            pid = self.images_by_id.keys()[0]
        else:
            raise Exception("Cannot resolve image to plate")
        return self.images_by_id[pid][iid].name.val

    def parse_plate(self, plate, wells_by_location, wells_by_id, images_by_id):
        """
        Accepts PlateData instances
        """
        # TODO: This should use the PlateNamingConvention. We're assuming rows
        # as alpha and columns as numeric.
        for well in plate.wells:
            wells_by_id[well.id.val] = well
            row = well.row.val
            # 0 offsetted is not what people use in reality
            column = str(well.column.val + 1)
            try:
                columns = wells_by_location[self.AS_ALPHA[row]]
            except KeyError:
                wells_by_location[self.AS_ALPHA[row]] = columns = dict()
            columns[column] = well

            for well_sample in well.well_samples:
                image = well_sample.image
                images_by_id[image.id.val] = image
        log.debug('Completed parsing plate: %s' % plate.name.val)
        for row in wells_by_location:
            log.debug('%s: %r' % (row, wells_by_location[row].keys()))

    def resolve_well(self, column, row, value):
            m = self.WELL_REGEX.match(value)
            if m is None or len(m.groups()) != 2:
                msg = 'Cannot parse well identifier "%s" from row: %r'
                msg = msg % (value, [o[1] for o in row])
                raise MetadataError(msg)
            plate_row = m.group(1).lower()
            plate_column = str(long(m.group(2)))
            wells_by_location = None
            if len(self.wells_by_location) == 1:
                wells_by_location = self.wells_by_location.values()[0]
                log.debug(
                    'Parsed "%s" row: %s column: %s' % (
                        value, plate_row, plate_column))
            else:
                for column, plate in row:
                    if column.__class__ is PlateColumn:
                        wells_by_location = self.wells_by_location[plate]
                        log.debug(
                            'Parsed "%s" row: %s column: %s plate: %s' % (
                                value, plate_row, plate_column, plate))
                        break
            if wells_by_location is None:
                raise MetadataError(
                    'Unable to locate Plate column in Row: %r' % row
                )
            try:
                return wells_by_location[plate_row][plate_column].id.val
            except KeyError:
                log.debug('Row: %s Column: %s not found!' % (
                    plate_row, plate_column))
                return -1L


class ScreenValueResolver(SPWValueResolver):
    """
    Value resolver for column types which is responsible for filling up
    non-metadata columns with their OMERO data model identifiers.
    """
    def __init__(self, data_retriever, target_object):
        super(ScreenValueResolver, self).__init__(data_retriever, target_object)
        self._load()

    def get_plate_name_by_id(self, plate):
        plate = self.plates_by_id[plate]
        return plate.name.val

    def get_well_by_id(self, well_id, plate=None):
        wells = self.wells_by_id[plate]
        return wells[well_id]

    def get_well_name(self, well_id, plate=None):
        well = self.get_well_by_id(well_id, plate)
        row = well.row.val
        col = well.column.val
        row = self.AS_ALPHA[row]
        return '%s%d' % (row, col + 1)

    def resolve_plate(self, value):
        try:
            return self.plates_by_name[value].id.val
        except KeyError:
            log.warn('Screen is missing plate: %s' % value)
            return Skip()

    def _load(self):
        self.target_object = \
            self.data_retriever.get_screen_info(self.target_object.id.val)
        self.target_name = unwrap(self.target_object.getName())
        self.images_by_id = dict()
        self.wells_by_location = dict()
        self.wells_by_id = dict()
        self.plates_by_name = dict()
        self.plates_by_id = dict()
        images_by_id = dict()
        self.images_by_id[self.target_object.id.val] = images_by_id
        for plate in (l.child for l in self.target_object.copyPlateLinks()):
            plate = self.data_retriever.get_plate(plate.id.val)
            self.plates_by_name[plate.name.val] = plate
            self.plates_by_id[plate.id.val] = plate
            wells_by_location = dict()
            wells_by_id = dict()
            self.wells_by_location[plate.name.val] = wells_by_location
            self.wells_by_id[plate.id.val] = wells_by_id
            self.parse_plate(
                PlateData(plate), wells_by_location, wells_by_id, images_by_id
            )



class PlateValueResolver(SPWValueResolver):
    """
    Value resolver for column types which is responsible for filling up
    non-metadata columns with their OMERO data model identifiers.
    """
    def __init__(self, data_retriever, target_object):
        super(PlateValueResolver, self).__init__(data_retriever, target_object)
        self._load()

    def get_well_by_id(self, well_id, plate=None):
        plate = self.target_object.id.val
        wells = self.wells_by_id[plate]
        return wells[well_id]

    def _load(self):
        self.target_object = \
            self.data_retriever.get_plate(self.target_object.id.val)
        self.target_name = unwrap(self.target_object.getName())
        self.wells_by_location = dict()
        self.wells_by_id = dict()
        wells_by_location = dict()
        wells_by_id = dict()

        self.images_by_id = dict()
        images_by_id = dict()

        self.wells_by_location[self.target_object.name.val] = wells_by_location
        self.wells_by_id[self.target_object.id.val] = wells_by_id
        self.images_by_id[self.target_object.id.val] = images_by_id
        self.parse_plate(
            PlateData(self.target_object),
            wells_by_location, wells_by_id, images_by_id
        )



class ProjectValueResolver(ValueResolver):
    """
    Value resolver for column types which is responsible for filling up
    non-metadata columns with their OMERO data model identifiers.
    """
    def __init__(self, data_retriever, target_object):
        super(ProjectValueResolver, self).__init__(data_retriever, target_object)
        self.images_by_id = defaultdict(lambda: dict())
        self.images_by_name = defaultdict(lambda: dict())
        self.datasets_by_id = dict()
        self.datasets_by_name = dict()
        self._load()

    def get_image_id_by_name(self, iname, dname=None):
        return self.images_by_name[dname][iname].id.val

    def get_image_name_by_id(self, iid, did=None):
        return self.images_by_id[did][iid].name.val

    def resolve_dataset(self, column, row, value):
        try:
            return self.datasets_by_name[value].id.val
        except KeyError:
            log.warn('Project is missing dataset: %s' % value)
            return Skip()

    def _load(self):
        project_id = self.target_object.id.val
        self.target_object = \
            self.data_retriever.get_project(project_id)
        self.target_name = self.target_object.name.val

        data = \
            self.data_retriever.get_datasets_and_images_for_project(project_id)

        seen = dict()
        for dataset, image in data:
            did = dataset.id.val
            dname = dataset.name.val
            iid = image.id.val
            iname = image.name.val
            log.info("Adding dataset:%d image:%s" % (did, iid))
            if dname in seen and seen[dname] != did:
                raise Exception("Duplicate datasets: '%s' = %s, %s" % (
                    dname, seen[dname], did
                ))
            else:
                seen[dname] = did

            ikey = (did, iname)
            if ikey in seen and iid != seen[ikey]:
                raise Exception("Duplicate image: '%s' = %s, %s (Dataset:%s)"
                                % (iname, seen[ikey], iid, did))
            else:
                seen[ikey] = iid

            self.images_by_id[did][iid] = image
            self.images_by_name[did][iname] = image
            self.datasets_by_id[did] = dataset
            self.datasets_by_name[dname] = dataset
        log.debug('Completed parsing project: %s' % self.target_object.id.val)



class DatasetValueResolver(ValueResolver):
    """
    Value resolver for column types which is responsible for filling up
    non-metadata columns with their OMERO data model identifiers.
    """
    def __init__(self, data_retriever, target_object):
        super(DatasetValueResolver, self).__init__(data_retriever, target_object)
        self.images_by_id = dict()
        self.images_by_name = dict()
        self._load()

    def get_image_id_by_name(self, iname, dname=None):
        return self.images_by_name[iname].id.val

    def get_image_name_by_id(self, iid, did):
        return self.images_by_id[did][iid].name.val

    def _load(self):
        dataset_id = self.target_object.id.val
        self.target_object = self.data_retriever.get_dataset(dataset_id)
        self.target_name = self.target_object.name.val

        data = self.data_retriever.get_image_links_for_dataset(dataset_id)

        images_by_id = dict()
        for image in data:
            iname = image.name.val
            iid = image.id.val
            images_by_id[iid] = image
            if iname in self.images_by_name:
                raise Exception("Image named %s(id=%d) present. (id=%s)" % (
                    iname, self.images_by_name[iname], iid
                ))
            self.images_by_name[iname] = image
        self.images_by_id[self.target_object.id.val] = images_by_id
        log.debug('Completed parsing dataset: %s' % self.target_name)
