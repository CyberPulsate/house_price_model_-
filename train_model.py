import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Adatok betöltése
train_data = pd.read_csv('train.csv') #csv fájl beolvasása

# Célváltozó és jellemzők kiválasztása
X = train_data.drop(columns=['SalePrice']) #Eltávolítjuk az oszlopot
y = train_data['SalePrice'] #Létrehozzuk a célváltozót

# Numerikus és kategorikus oszlopok kiválasztása
num_cols = X.select_dtypes(include=['int64', 'float64']).columns #Egész és lebegőpontos számok kiválasztása
cat_cols = X.select_dtypes(include=['object']).columns #Stringek kiválasztása

# Adatfeldolgozás, pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), #A hiányzó adatok helyére átlagot számítunk
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),#Pótoljuk a hiányzó értékeket
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer( #összeállítjuk az oszlopokat. 
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))#döntési fa összeállítása
])

# Adatok felosztása tanító és tesztelő halmazokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
''' 
X: Ez az input feature mátrix, amely tartalmazza az összes bemeneti változót (független változókat), amelyek alapján a modell tanulni fog.
y: Ez a célváltozó vektor, amely tartalmazza a célváltozókat, amiket a modell prediktálni fog.
test_size: Ez a paraméter azt határozza meg, hogy milyen arányban legyen felosztva az eredeti adathalmaz a teszt adathalmazba. Például test_size=0.2 azt jelenti, hogy az adatok 20%-a kerül a teszt adathalmazba, és 80%-a a tanító adathalmazba.
andom_state: Ez a paraméter meghatározza a véletlenszám-generátor kezdőállapotát, ami befolyásolja az adatok felosztásának véletlenszerűségét. Ha megadunk egy konkrét számot (pl. random_state=42), akkor az adatok minden futtatás során ugyanúgy fognak felosztódni, ami segíti a reprodukálhatóságot.
'''
# Pipeline betanítása
pipeline.fit(X_train, y_train)

# Modell mentése
joblib.dump(pipeline, 'house_price_model.pkl')
print("Model trained and saved as house_price_model.pkl")

#REST API
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Flask alkalmazás létrehozása
app = Flask(__name__)

# Modell betöltése
model = joblib.load('house_price_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame([data])
    prediction = model.predict(data_df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__': #Ez a kifejezés biztosítja, hogy az alkalmazás csak akkor induljon el, ha közvetlenül futtatjuk a fájlt
    app.run(debug=True) #Ez a parancs indítja el a Flask fejlesztői szerverét. debug=True beállítás segítségével bekapcsoljuk a fejlesztői mód funkciót, ami lehetővé teszi számunkra, hogy lássuk a hibákat