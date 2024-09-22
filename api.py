from pydantic.dataclasses import dataclass
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import Path
from fastapi import HTTPException
from sklearn.metrics import confusion_matrix
import mlflow                                              

ZIP_TEST_DATA_FILENAME = "test_data.zip"                
MLFLOW_MODEL_FOLDER = "mlflow_model"          
BEST_THRESHOLD = 0.27                                   

@dataclass
class Client_credit:
    SK_ID_CURR: int             # l'index du credit
    FLAG_OWN_REALTY: int             # Si le client est propriétaire de son logement
    FLAG_OWN_CAR: int             # Si le client a et est propriétaire d'une voiture
    OWN_CAR_AGE: float           # Age de la voiture
    NAME_INCOME_TYPE_Working: bool            # True si les revennus du client proviennent d'un salaire
    DAYS_EMPLOYED: float           # Ancienneté du client dans son emploi actuel
    AMT_GOODS_PRICE: float           # Prix du bien que le client veut acheter
    AMT_CREDIT: float           # Montant du prêt
    EXT_SOURCE_1_x: float           # score from external data source
    EXT_SOURCE_2_x: float           # score from external data source
    EXT_SOURCE_3_x: float           # score from external data source
    PRED_PROBA: float = 0       # probabilité que le client ait des retards de paiement
    PRED_TARGET: int = 0         # TARGET predite et calculé à partir de PRED_PROBA
    TARGET: bool = 0        # 0 si le client a des retards de paiement => crédit à rejeter

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
    SK_ID_CURR: int             # l'index du crédit
    FLAG_OWN_REALTY: int             # Si le client est propriétaire de son logement
    FLAG_OWN_CAR: int             # Si le client a et est propriétaire d'une voiture
    OWN_CAR_AGE: float           # Age de la voiture
    NAME_INCOME_TYPE_Working:   bool            # True si les revennus du client proviennent d'un salaire
    DAYS_EMPLOYED: float           # Ancienneté du client dans son emploi actuel
    AMT_GOODS_PRICE: float           # Prix du bien que le client veut acheter
    AMT_CREDIT: float           # Montant du prêt
    EXT_SOURCE_1_x: float           # score from external data source
    EXT_SOURCE_2_x: float           # score from external data source
    EXT_SOURCE_3_x: float           # score from external data source

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


print("\n   ")
print("Chargement du modèle pré-entraîné...")
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

print("\nInitialisation de l'API...")
app = FastAPI(debug=True)

@app.get("/get_client/{SK_ID_CURR}")
def get_client_by_ID(SK_ID_CURR: int = Path(ge=min_SK_ID_CURR, le=max_SK_ID_CURR)) -> Client_credit:
    part_data_df = merged_data_df[merged_data_df['SK_ID_CURR'] == SK_ID_CURR]
    if part_data_df.shape[0] == 0:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")
    return Client_credit_from_data(part_data_df)

@app.post("/post_client/{Client_new_credit}")
def calcul_nouveau_credit(new_client: Client_new_credit) -> Client_credit:
    part_data_df = merged_data_df[merged_data_df['SK_ID_CURR'] == new_client.SK_ID_CURR]
    if part_data_df.shape[0] == 0:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")
    return Client_credit_from_data(to_Client_credit(new_client).to_new_data())

@app.get("/matrice_confusion")
def matrice_confusion() -> dict:
    return {"Matrice de confusion": f"TN={tn} - FN={fn} - FP={fp} - TP={tp}",
            "Quelques TN": ", ".join(map(str, merged_data_df[(merged_data_df['y_pred'] == 1) & (merged_data_df['TARGET'] == 0)]['SK_ID_CURR'].sample(10).to_list())),
            "Quelques FN": ", ".join(map(str, merged_data_df[(merged_data_df['y_pred'] == 0) & (merged_data_df['TARGET'] == 1)]['SK_ID_CURR'].sample(10).to_list())),
            "Quelques FP": ", ".join(map(str, merged_data_df[(merged_data_df['y_pred'] == 0) & (merged_data_df['TARGET'] == 0)]['SK_ID_CURR'].sample(10).to_list())),
            "Quelques TP": ", ".join(map(str, merged_data_df[(merged_data_df['y_pred'] == 1) & (merged_data_df['TARGET'] == 1)]['SK_ID_CURR'].sample(10).to_list()))}
