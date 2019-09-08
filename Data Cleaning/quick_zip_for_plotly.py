# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:34:28 2019

@author: imcna
"""
import pandas as pd

from uszipcode import SearchEngine
def to_lng(zipcode):
    search = SearchEngine(simple_zipcode=True)
    lng = search.by_zipcode(zipcode).to_dict()['lng']
    return lng
def to_lat(zipcode):
    search = SearchEngine(simple_zipcode=True)
    lat = search.by_zipcode(zipcode).to_dict()['lat']
    return lat

dataset = pd.read_csv('pct_zips.csv')

# Zipcodes to Distance
dataset['lng'] = dataset['zipcode'].apply(to_lng)
dataset['lat'] = dataset['zipcode'].apply(to_lat)

dataset.to_csv('Discrete_LocationTurnoverPct_Counts.csv', index = False)