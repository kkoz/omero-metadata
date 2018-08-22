
import logging
import re

from omero.grid import ImageColumn, LongColumn, PlateColumn, RoiColumn
from omero.grid import StringColumn, WellColumn, DoubleColumn, BoolColumn
from omero.grid import DatasetColumn

PLATE_NAME_COLUMN = 'Plate Name'
WELL_NAME_COLUMN = 'Well Name'
DATASET_NAME_COLUMN = 'Dataset Name'
IMAGE_NAME_COLUMN = 'Image Name'

COLUMN_TYPES = {
    'plate': PlateColumn, 'well': WellColumn, 'image': ImageColumn,
    'dataset': DatasetColumn, 'roi': RoiColumn,
    'd': DoubleColumn, 'l': LongColumn, 's': StringColumn, 'b': BoolColumn
}

DEFAULT_COLUMN_SIZE = 1

REGEX_HEADER_SPECIFIER = r'# header '

log = logging.getLogger("omero.util.populate_metadata")

def parse_column_types(column_type_list):
    column_types = []
    for column_type in column_type_list:
        if column_type.lower() in COLUMN_TYPES:
            column_types.append(column_type.lower())
        else:
            column_types = []
            message = "\nColumn type '%s' unknown.\nChoose from following: " \
                "%s" % (column_type, ",".join(COLUMN_TYPES.keys()))
            raise MetadataError(message)
    return column_types


class HeaderResolver(object):
    """
    Header resolver for known header names which is responsible for creating
    the column set for the OMERO.tables instance.
    """

    def __init__(self, logging_level="INFO"):
        logging.basicConfig(level=logging_level)
        return

    @staticmethod
    def is_row_column_types(row):
        if "# header" in row[0]:
            return True
        return False

    @staticmethod
    def get_column_types(row):
        if "# header" not in row[0]:
            return None
        get_first_type = re.compile(REGEX_HEADER_SPECIFIER)
        column_types = [get_first_type.sub('', row[0])]
        for column in row[1:]:
            column_types.append(column)
        column_types = parse_column_types(column_types)
        return column_types

    def get_keys(self):
        #To be implemented by subclasses
        pass

    def columns_sanity_check(self, columns):
        column_types = [column.__class__ for column in columns]
        if WellColumn in column_types and ImageColumn in column_types:
            log.debug(column_types)
            raise MetadataError(
                ('Well Column and Image Column cannot be resolved at '
                 'the same time. Pick one.'))
        log.debug('Sanity check passed')

    def create_columns(self, headers, column_types):
        if column_types is not None and len(column_types) != len(headers):
            message = "Number of columns and column types not equal."
            raise MetadataError(message)
        columns = list()
        headers_as_lower = [v.lower() for v in headers]
        for i, header_as_lower in enumerate(headers_as_lower):
            name = headers[i]
            description = ""
            if "%%" in name:
                name, description = name.split("%%", 1)
                name = name.strip()
                # description is key=value. Convert to json
                if "=" in description:
                    k, v = description.split("=", 1)
                    k = k.strip()
                    description = json.dumps({k: v.strip()})
            # HDF5 does not allow / in column names
            name = name.replace('/', '\\')
            if column_types is not None and \
                    COLUMN_TYPES[column_types[i]] is StringColumn:
                column = COLUMN_TYPES[column_types[i]](
                    name, description, DEFAULT_COLUMN_SIZE, list())
            elif column_types is not None:
                column = COLUMN_TYPES[column_types[i]](name, description, list())
            else:
                try:
                    keys = self.get_keys()
                    log.debug("Adding keys %r" % keys)
                    if keys[header_as_lower] is StringColumn:
                        column = keys[header_as_lower](
                            name, description,
                            DEFAULT_COLUMN_SIZE, list())
                    else:
                        column = keys[header_as_lower](
                            name, description, list())
                except KeyError:
                    log.debug("Adding string column %r" % name)
                    column = StringColumn(
                        name, description, DEFAULT_COLUMN_SIZE, list())
            log.debug("New column %r" % column)
            columns.append(column)
        append = []
        for column in columns:
            if column.__class__ is PlateColumn:
                append.append(StringColumn(PLATE_NAME_COLUMN, '',
                              DEFAULT_COLUMN_SIZE, list()))
            if column.__class__ is WellColumn:
                append.append(StringColumn(WELL_NAME_COLUMN, '',
                              DEFAULT_COLUMN_SIZE, list()))
#            if column.__class__ is ImageColumn:
#                append.append(StringColumn(IMAGE_NAME_COLUMN, '',
#                              DEFAULT_COLUMN_SIZE, list()))
            # Currently hard-coded, but "if image name, then add image id"
            if column.name == IMAGE_NAME_COLUMN:
                append.append(ImageColumn("Image", '', list()))
        columns.extend(append)
        self.columns_sanity_check(columns)
        return columns


class DatasetHeaderResolver(HeaderResolver):

    dataset_keys = {
        'image': ImageColumn,
        'image_name': StringColumn,
    }

    def get_keys(self):
        return self.dataset_keys


class ProjectHeaderResolver(HeaderResolver):
    project_keys = {
        'dataset': StringColumn,  # DatasetColumn
        'dataset_name': StringColumn,
        'image': ImageColumn,
        'image_name': StringColumn,
    }

    def get_keys(self):
        return self.project_keys

class PlateHeaderResolver(HeaderResolver):

    plate_keys = dict({
        'well': WellColumn,
        'field': ImageColumn,
        'row': LongColumn,
        'column': LongColumn,
        'wellsample': ImageColumn,
        'image': ImageColumn,
    })

    def get_keys(self):
        return self.plate_keys

class ScreenHeaderResolver(HeaderResolver):

    screen_keys = dict({
        'plate': PlateColumn,
        'well': WellColumn,
        'field': ImageColumn,
        'row': LongColumn,
        'column': LongColumn,
        'wellsample': ImageColumn,
        'image': ImageColumn
    })

    def get_keys(self):
        return self.screen_keys


