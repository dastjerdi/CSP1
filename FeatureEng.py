import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import concurrent.futures
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import time
from sklearn.linear_model import LassoCV




"""
Read in train and test as Pandas DataFrames
"""
print("read data")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


i = 0
def getBin(smile):
    global i
    i += 1
    print(i)
    mol = Chem.MolFromSmiles(smile)
    fprint = Chem.rdmolops.RDKFingerprint(mol)
    return list(map(int, fprint.ToBitString()))

with concurrent.futures.ProcessPoolExecutor() as executor:
    Bits_test = executor.map(getBin, df_test['smiles'], chunksize=50)
    Bits_train = executor.map(getBin, df_train['smiles'].iloc[1:100000], chunksize=50)


Y_train = df_train.gap.values.iloc[1:100000]
start = time.time()
X_train = pd.DataFrame(Bits_train)
X_test= pd.DataFrame(Bits_test)
end = time.time()
print(end-start)

print("Begin PCA")
pca = PCA(n_components = 10)
pca.fit(X_train, Y_train)

X_PCA = pca.transform(X_train)

X_PCA_Test = pca.transform(X_test)

print("begin lasso CV")
Lasso = LassoCV()
Lasso.fit(X_PCA, Y_train)
Lasso_pred = L

print("Lasso CV done")
def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("Preds_lasso.csv", Lasso.predict(X_PCA_Test))
print("File made!")

print("begin RF CV")
forest = RandomForestRegressor(n_estimators = 100, max_depth = 5)
forest.fit(X_PCA, Y_train)

print("RF CV done")
def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("Preds_rf.csv", forest.predict(X_PCA_Test))
print("File made!")
