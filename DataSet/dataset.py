import pandas as pd

class DataSetPolyps():
    #constructor
    def __init__(self,nombre):
        #el str.upper es para convertir todo a mayuscula
        if (str.upper(nombre)=='WL'):
            #cargo el dataframe para WL
            #self apuntador para los atributos
            # __ es privado
            self.__trainDataSet = pd.read_csv('DataSet/trainWl.csv')
            columns = ['x_col', 'y_col']
            self.__trainDataSet.columns = columns

            self.__testDataSet = pd.read_csv('DataSet/testWl.csv')
            columns = ['x_col', 'y_col']
            self.__testDataSet.columns = columns
        else:
            self.__trainDataSet = pd.read_csv('DataSet/trainNbi.csv')
            columns = ['x_col', 'y_col']
            self.__trainDataSet.columns = columns

            self.__testDataSet = pd.read_csv('DataSet/testNbi.csv')
            columns = ['x_col', 'y_col']
            self.__testDataSet.columns = columns
    
    def getTrainDataSet(self):
        return self.__trainDataSet

    def getTestDataSet(self):
        return self.__testDataSet

