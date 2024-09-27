from pydantic.dataclasses import dataclass
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import Path
from fastapi import HTTPException
from sklearn.metrics import confusion_matrix
import mlflow                                              

from fastapi import FastAPI
from test_api import app as test_api

main_app = FastAPI()

@main_app.get("/test_api")
async def redirect_to_test_api():
    return {"message": "Redirect to test API"}

main_app.mount("/test_api", test_api)

ZIP_TEST_DATA_FILENAME = "test_data.zip"                
MLFLOW_MODEL_FOLDER = "mlflow_model"          
BEST_THRESHOLD = 0.27                                   

@dataclass
class Client_credit:
    SK_ID_CURR: int           
    FLAG_OWN_REALTY: int             
    FLAG_OWN_CAR: int             
    OWN_CAR_AGE: float         
    NAME_INCOME_TYPE_Working: bool           
    DAYS_EMPLOYED: float        
    AMT_GOODS_PRICE: float          
    AMT_CREDIT: float          
    EXT_SOURCE_1_x: float          
    EXT_SOURCE_2_x: float          
    EXT_SOURCE_3_x: float           
    PRED_PROBA: float = 0     
    PRED_TARGET: int = 0         
    TARGET: bool = 0       

    def to_new_data(self):
        new_data_df = merged_data_df[merged_data_df['SK_ID_CURR'] == self.SK_ID_CURR].copy() 
        new_data_df['FLAG_OWN_REALTY'] = self.FLAG_OWN_REALTY
        new_data_df['FLAG_OWN_CAR'] = self.FLAG_OWN_CAR
        new_data_df['OWN_CAR_AGE'] = self.OWN_CAR_AGE
        new_data_df['NAME_INCOME_TYPE_Working'] = self.NAME_INCOME_TYPE_Working
        new_data_df['DAYS_EMPLOYED'] = self.DAYS_EMPLOYED
        new_data_df['AMT_GOODS_PRICE'] = self.AMT_GOODS_PRICE
        new_data_df['AMT_CREDIT'] = self.AMT_CREDIT
        new_data_df['EXT_SOURCE_1_x'] = self.EXT_SOURCE_1_x
        new_data_df['EXT_SOURCE_2_x'] = self.EXT_SOURCE_2_x
        new_data_df['EXT_SOURCE_3_x'] = self.EXT_SOURCE_3_x
        new_X = new_data_df.drop(columns=['TARGET', 'y_pred_proba', 'y_pred'])
        new_y_proba = model.predict_proba(new_X)[:, 1][0]
        new_y_pred = int(np.where(new_y_proba >= BEST_THRESHOLD, 1, 0))
        new_data_df['y_pred_proba'] = new_y_proba
        new_data_df['y_pred'] = new_y_pred
        return new_data_df

@dataclass
class Client_new_credit:
    SK_ID_CURR: int            
    FLAG_OWN_REALTY: int            
    FLAG_OWN_CAR: int           
    OWN_CAR_AGE: float        
    NAME_INCOME_TYPE_Working:   bool           
    DAYS_EMPLOYED: float           
    AMT_GOODS_PRICE: float          
    AMT_CREDIT: float          
    EXT_SOURCE_1_x: float           
    EXT_SOURCE_2_x: float           
    EXT_SOURCE_3_x: float           

def Client_credit_from_data(client_data) -> Client_credit:
    return Client_credit(SK_ID_CURR = client_data['SK_ID_CURR'],
                         FLAG_OWN_REALTY = client_data['FLAG_OWN_REALTY'],
                         FLAG_OWN_CAR = client_data['FLAG_OWN_CAR'],
                         OWN_CAR_AGE = client_data['OWN_CAR_AGE'],
                         NAME_INCOME_TYPE_Working = client_data['NAME_INCOME_TYPE_Working'],
                         DAYS_EMPLOYED = client_data['DAYS_EMPLOYED'],
                         AMT_GOODS_PRICE = client_data['AMT_GOODS_PRICE'],
                         AMT_CREDIT = client_data['AMT_CREDIT'],
                         EXT_SOURCE_1_x = client_data['EXT_SOURCE_1_x'],
                         EXT_SOURCE_2_x = client_data['EXT_SOURCE_2_x'],
                         EXT_SOURCE_3_x = client_data['EXT_SOURCE_3_x'],
                         PRED_PROBA = client_data['y_pred_proba'],
                         PRED_TARGET = client_data['y_pred'],
                         TARGET = client_data['TARGET'])
    
def to_Client_credit(credit: Client_new_credit) -> Client_credit:
    return Client_credit(credit.SK_ID_CURR, credit.FLAG_OWN_REALTY, credit.FLAG_OWN_CAR,
                         credit.OWN_CAR_AGE, credit.NAME_INCOME_TYPE_Working, credit.DAYS_EMPLOYED,
                         credit.AMT_GOODS_PRICE, credit.AMT_CREDIT, credit.EXT_SOURCE_1_x,
                         credit.EXT_SOURCE_2_x, credit.EXT_SOURCE_3_x)

model = mlflow.sklearn.load_model("mlflow_model")
print("Chargement des données de test...")
temp_df  = pd.read_csv(ZIP_TEST_DATA_FILENAME, sep=',', encoding='utf-8', compression='zip')
min_SK_ID_CURR  = temp_df['SK_ID_CURR'].min()
max_SK_ID_CURR  = temp_df['SK_ID_CURR'].max()
y = temp_df['TARGET']
X = temp_df.drop(columns='TARGET')
y_pred_proba   = model.predict_proba(X)[:, 1]
y_pred         = np.where(y_pred_proba >= BEST_THRESHOLD, 1, 0)
merged_data_df = pd.concat([temp_df, pd.DataFrame(y_pred_proba, columns=['y_pred_proba']), pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
del temp_df
del y_pred_proba
del y_pred
app = FastAPI(debug=True)

@app.get("/get_client/{SK_ID_CURR}")
def get_client_by_ID(SK_ID_CURR: int = Path(ge=min_SK_ID_CURR, le=max_SK_ID_CURR)) -> Client_credit:
    part_data_df = merged_data_df[merged_data_df['SK_ID_CURR'] == SK_ID_CURR]
    if part_data_df.shape[0] == 0:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")
    return Client_credit_from_data(part_data_df)
