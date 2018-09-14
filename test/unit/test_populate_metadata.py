#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import omero.model
import omero.grid
import omero.clients

from omero.rtypes import rlong, rlist, rstring, unwrap
from omero.model import DatasetAnnotationLinkI, DatasetI, FileAnnotationI
from omero.model import OriginalFileI, PlateI, PlateAnnotationLinkI, ScreenI
from omero.model import PlateAcquisitionI, WellI, WellSampleI, ImageI
from omero.model import ProjectAnnotationLinkI, ProjectI
from omero.model import ScreenAnnotationLinkI
from omero.model import MapAnnotationI, NamedValue
from omero.grid import ImageColumn, LongColumn, PlateColumn, RoiColumn
from omero.grid import StringColumn, WellColumn, DoubleColumn, BoolColumn
from omero.grid import DatasetColumn


from populate_metadata import *
from value_resolvers import *
from header_resolvers import *

###Generic Mocks###
class MockParent(object):
    def __init__(self, child):
        self.child = child

class MockValued(object):
    def __init__(self, val):
        self.val = val

class MockIdentifiable(object):
    def __init__(self, val):
        self.id = MockValued(val)
        return

class MockNamedIdentifiable(object):
    def __init__(self, id_val, name_val):
        self.id = MockValued(id_val)
        self.name = MockValued(name_val)


###Mock Omero Connection###
class MockSession(object):
    def __init__(self):
        self.mock_query_service = MockQueryService()
        return

    def getQueryService(self):
        return self.mock_query_service

class MockQueryService():
    def projection(self, a, b, c):
        return [[rlong(0)]]

    def findByQuery(self, a, b, c):
        return parse_target_object("Plate:1")

class MockClient(object):
    def __init__(self):
        self.sf = MockSession()
        return

    def getSession(self):
        return self.sf

###Mock Populate Metadata Classes###
class MockWellSample(MockIdentifiable):
    def __init__(self, well_sample_id, image_id):
        super(MockWellSample, self).__init__(well_sample_id)
        self.image = MockNamedIdentifiable(image_id, "image %s" % (image_id))
    def getImage(self):
        return self.image


class MockWell(object):
    def __init__(self, id_val, row, col, well_samples):
        self.id = MockValued(id_val)
        self.row = MockValued(row)
        self.column = MockValued(col)
        self.well_samples = well_samples
    def copyWellSamples(self):
        return self.well_samples

class MockPlate(MockNamedIdentifiable):
    def __init__(self, id_val, name_val, wells):
        super(MockPlate, self).__init__(id_val, name_val)
        self.wells = wells
    def getName(self):
        return rstring(self.name.val)
    def copyWells(self):
        return self.wells


dummy_plates = []
for i in range(0, 3):
    #Create Wells
    wells = []
    for j in range(0, 4):
        well_samples = [MockWellSample(str(k), str(k)) for
                        k in range(i*100 + j*10, i*100 + j*10 +3)]
        id_num = i*10 + j
        row_num = i*4 + j
        well = MockWell(str(id_num), row_num, j%2, well_samples)
        wells.append(well)
    plate = MockPlate(str(i), "plate %d" % (i), wells)
    dummy_plates.append(plate)


class MockScreenData(object):
    def __init__(self):
        self.id = MockValued("1")
            
            
    def getName(self):
        return rstring("Name")

    def copyPlateLinks(self):
        links = [MockParent(dp) for dp in dummy_plates]
        return links

class MockDataRetriever(object):
    def __init__(self):
        return

    def get_target_group(self, target_object):
        return {}

    def get_screen_info(self, screen_id):
        return MockScreenData()

    def get_plate(self, plate_id):
        return dummy_plates[int(plate_id)]


class MockValueResolver():
    AS_ALPHA = [chr(v) for v in range(97, 122 + 1)]  # a-z
    # Support more than 26 rows
    for v in range(97, 122 + 1):
        AS_ALPHA.append('a' + chr(v))
    WELL_REGEX = re.compile(r'^([a-zA-Z]+)(\d+)$')

    def __init__(self, target_id="1"):
        self.data_retriever = MockDataRetriever()
        self.target_object = MockIdentifiable(target_id)
        return

    def resolve(self, column, value, row):
        return value

class MockHeaderResolver():
    def __init__(self):
        return

class MockParsingUtilFactory():
    def __init__(self):
        self.value_resolver = MockValueResolver()
        self.header_resolver = MockHeaderResolver()
    
    def get_value_resolver(self):
        return self.value_resolver

    def get_header_resolver(self):
        return self.header_resolver

mock_client = MockClient()
mock_parsing_util_factory = MockParsingUtilFactory()

def get_dummy_string_columns():
    dummy_columns = [
        StringColumn("StrCol1", "1st Str Col", 1, list()),
        StringColumn("StrCol2", "2nd Str Col", 1, list())]
    return  dummy_columns

def get_dummy_double_columns():
    dummy_columns = [
        DoubleColumn("DoubleCol1", "1st Double Col", 1, list()),
        DoubleColumn("DoubleCol2", "2nd Double Col", 1, list())]
    return  dummy_columns


def get_dummy_string_rows():
    dummy_string_rows = [["Test Val 1.1", "Test Val 1.2"],
                     ["Test Val 2.1", "Test Val 2.2"]]
    return dummy_string_rows

string_column_types = ['s','s']

def get_dummy_mixed_rows():
    dummy_mixed_rows = [["Test Val 1.1", "2.2"]]

dummy_rows_wrong_size = [["a", "b", "c"]]

def test_get_column_types():
    return

def test_populate_row():
    ctx = ParsingContext(
        mock_client,
        parse_target_object("Plate:1"),
        mock_parsing_util_factory,
        None,
        column_types=string_column_types)

    #We setup the columns manually
    ctx.columns = get_dummy_string_columns()

    dummy_string_rows = get_dummy_string_rows()
    ctx.populate_row(dummy_string_rows[0])
    for i, c in enumerate(ctx.columns):
        for j, v in enumerate(c.values):
            assert dummy_string_rows[j][i] == v

def test_get_column_widths():
    ctx = ParsingContext(
        mock_client,
        parse_target_object("Plate:1"),
        mock_parsing_util_factory,
        None,
        column_types=string_column_types)   


def test_preprocess_data():
    ctx = ParsingContext(
        mock_client,
        parse_target_object("Plate:1"),
        mock_parsing_util_factory,
        None,
        column_types=string_column_types)
    dummy_string_rows = get_dummy_string_rows()
    #We setup the columns manually
    ctx.columns = get_dummy_string_columns()

    #Test string column size increase
    ctx.preprocess_data(get_dummy_string_rows())
    widths = ctx.get_column_widths()
    for width in widths:
        assert width == len(get_dummy_string_rows()[0][0])

###Header Resolver Tests###
def test_header_resolvers():
    header_resolver = HeaderResolver()

def test_is_row_column_types():
    assert HeaderResolver.is_row_column_types(["test"]) == False
    assert HeaderResolver.is_row_column_types(["# header"]) == True
    with pytest.raises(IndexError):
        HeaderResolver.is_row_column_types([])

###Test Value Resolvers###

def test_spw_value_resovler():
    plate_num = 0
    spw = SPWValueResolver(MockDataRetriever(), MockIdentifiable(str(plate_num)))
    wells_by_location = dict()
    wells_by_id = dict()
    images_by_id = dict()
    spw.parse_plate(dummy_plates[plate_num], wells_by_location, wells_by_id, images_by_id)
    for well in dummy_plates[plate_num].wells:
        well_id = well.id.val
        assert wells_by_id[well_id] == well
        row = spw.AS_ALPHA[well.row.val]
        col = str(well.column.val + 1)
        assert wells_by_location[row][col] == well
        for well_sample in well.well_samples:
            assert images_by_id[well_sample.image.id.val] == well_sample.image


def test_load_screen_value_resolver():
    sw = ScreenValueResolver(MockDataRetriever(), MockIdentifiable("1"))
    for plate in dummy_plates:
        assert sw.get_plate_name_by_id(plate.id.val) == plate.name.val
        assert plate.id.val == sw.resolve_plate(plate.name.val)
        for well in plate.wells:
            assert sw.get_well_by_id(well.id.val, plate.id.val) == well

def test_load_plate_value_resolver():
    test_id = "0"
    pw = PlateValueResolver(MockDataRetriever(), MockIdentifiable(test_id))
    for well in dummy_plates[int(test_id)].wells:
        assert WellData(well) == pw.get_well_by_id(well.id.val)
        
#def test_plate_vw_get_image_name_by_id():
#    test_id = "0"
#    pw = PlateValueResolver(MockDataRetriever(), MockIdentifiable(test_id))
#    for well in dummy_plates[int(test_id)].wells:
#        for well_sample in well.well_samples:
#            assert pw.get_image_name_by_id(well_sample.image.id.val, pid=test_id)
