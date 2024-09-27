import json
import requests
import pytest      

API_BASE_URL = "https://projet7-92ff04160aea.herokuapp.com" 
API_GET = "/get_client/"

HTTPS_TIMEOUT = 30                                                 
GOOD_ID_CLIENT = 456242
BAD_ID_CLIENT  = 207109
BAD_ID_FORMAT  = 45

GOOD_DICT_CLIENT = {"SK_ID_CURR": 456242,
                    "FLAG_OWN_REALTY": 1,
                    "FLAG_OWN_CAR": 1,
                    "OWN_CAR_AGE": 10,
                    "NAME_INCOME_TYPE_Working": false,
                    "DAYS_EMPLOYED": -3689,
                    "AMT_GOODS_PRICE": 1125000,
                    "AMT_CREDIT": 1312110,
                    "EXT_SOURCE_1_x": 0.5021298056566625,
                    "EXT_SOURCE_2_x": 0.746258730569961,
                    "EXT_SOURCE_3_x": 0.4066174366275036}

BAD_DICT_ID_CLIENT = {"SK_ID_CURR": 207109,
                      "FLAG_OWN_REALTY": 1,
                      "FLAG_OWN_CAR": 0,
                      "OWN_CAR_AGE": 12.061090818687727,
                      "NAME_INCOME_TYPE_Working": 0,
                      "DAYS_EMPLOYED": -768,
                      "AMT_GOODS_PRICE": 553500,
                      "AMT_CREDIT": 641173.5,
                      "EXT_SOURCE_1_x": 0.8427634659543568,
                      "EXT_SOURCE_2_x": 0.6816988025574287,
                      "EXT_SOURCE_3_x": 0.7544061731797895}

BAD_DICT_FORMAT = {"SK_ID_CURR": 207108,
                   "FLAG_OWN_REALTY": 1,
                   "FLAG_OWN_CAR": 0,
                   "OWN_CAR_AGE": 12.061090818687727,
                   "AMT_CREDIT": 641173.5,
                   "EXT_SOURCE_1_x": 0.8427634659543568,
                   "EXT_SOURCE_2_x": 0.6816988025574287,
                   "EXT_SOURCE_3_x": 0.7544061731797895}

def test_get():
    # Test un id qui n'est pas dans la fourchette des id du serveur
    result = requests.get(f"{API_BASE_URL}{API_GET}{BAD_ID_FORMAT}", timeout=HTTPS_TIMEOUT)
    assert result.status_code == 422, "Le serveur n'a pas répondu 422 comme attendu"

    # Test un id inconnu du serveur
    result = requests.get(f"{API_BASE_URL}{API_GET}{BAD_ID_CLIENT}", timeout=HTTPS_TIMEOUT)
    assert result.status_code == 404, "Le serveur n'a pas répondu 404 comme attendu"

    result = requests.get(f"{API_BASE_URL}{API_GET}{GOOD_ID_CLIENT}", timeout=HTTPS_TIMEOUT)
    assert result.status_code == 200, "Le serveur n'a pas répondu 200 comme attendu"

    # Decode et convertie le JSON format en dictionaire
    dict_obj = json.loads(result.content)
    assert len(dict_obj) == 14, "Le client retourné par le serveur doit contenir 14 valeurs"

test_get()
