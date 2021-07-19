from operator import index
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import pickle
import dalex as dx
from lime import lime_tabular
from lime import submodular_pick
from timeit import default_timer as timer
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from tensorflow.python.keras.backend import dtype

index_featurer_names = {}
bus=1
for i in range(77):
    if (i % 2 == 0):
        index_featurer_names[i] = ('bus_'+str(bus)+':u')
    else:
        index_featurer_names[i] = ('bus_'+str(bus)+':phi')
        bus += 1

class SmartGridExplainer:
    def __init__(self, X_train, X_test,y_train,y_test, model,grid):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.grid = grid
        feature_names = []
        bus = 1
        if(grid == 'NineBusSystem'):
            features = 17
        else:
            features = 77
        for i in range(features):
            if (i % 2 == 0):
                feature_names.append('bus_'+str(bus)+':u')
            else:
                feature_names.append('bus_'+str(bus)+':phi')
                bus += 1
        self.feature_names = feature_names
        
        # if(reduced==1):
        #     self.feature_names = feature_names[9:len(feature_names)]
        
        self.grid = grid
        # self.reduced=reduced
        
    def dnn_model_predict(self,x):
        
        
        feats = x.shape[1]
        d = {}
        for i in range(feats):
            d['F_%s' % i] = x[:,i]

        df = pd.DataFrame.from_dict(d)
        test_ds = df_to_singlerow(df, shuffle=False)
        result = self.model.predict(test_ds)
        
        stable = result
        unstable = 1 - result
        distribution = np.concatenate((unstable,stable),axis=1)
        print('distr: ', distribution.shape)
        return distribution
    
    def rnn_model_predict(self,x):
        reshaped_instance = np.reshape(x, (len(x),1,-1))

            
        result = np.squeeze(self.model.predict(reshaped_instance))
        print(result)
            
        result = np.reshape(result, (len(x),-1))
            
        stable = result
        unstable = 1 - result
        distribution = np.concatenate((unstable,stable),axis=1)
            
        return distribution
    
    #Generates a Local Lime Explanation explaning the instance i of the test set
    #Stores it in explainer outputs folder
    def generateLocalLimeExplanation(self,i):
        explainer = lime_tabular.LimeTabularExplainer(training_data= np.array(self.X_train),class_names=['unstable', 'stable'], mode="classification",feature_names=self.feature_names)
        model_name = type(self.model).__name__

        
        

        if(model_name == 'Sequential'):
            if(self.model.name == 'DNN'):
                predict_fn = self.dnn_model_predict
                model_name = 'DNN'
            elif(self.model.name == 'RNN'):
                predict_fn = self.rnn_model_predict
                model_name = 'RNN'
            start = timer()
            exp = explainer.explain_instance(data_row = np.squeeze(self.X_test[i]), predict_fn = predict_fn , num_features=17)
            end = timer()
        else:
            start = timer()
            exp = explainer.explain_instance(data_row = np.array(self.X_test.iloc[i]), predict_fn = self.model.predict_proba, num_features=17)
            end = timer()
        
        #Path to store Local Explanation Outputs
        dir_name = os.path.join('explainer_outputs','LIME','Local',self.grid)
        base_filename = model_name  + str(i)
        
        suffix = '.html'

        pathLocal = os.path.join(dir_name, base_filename + suffix)
        singleExplanationTime = end-start
        exp.save_to_file(pathLocal)
        print('Single Lime Explanation Time:', singleExplanationTime)

    def generateGlobalLimeExplanations(self):
        explainer = lime_tabular.LimeTabularExplainer(training_data= np.array(self.X_train),class_names=['unstable', 'stable'], mode="classification",feature_names=self.feature_names)
        model_name = type(self.model).__name__
       
        

        #LIME Global Explainer with Submodular Pick
        
        
        if(model_name == 'Sequential'):
            if(self.model.name == 'DNN'):
                predict_fn = self.dnn_model_predict
                model_name = 'DNN'
            elif(self.model.name == 'RNN'):
                predict_fn = self.rnn_model_predict
                model_name = 'RNN'
        else:
            predict_fn = self.model.predict_proba
        
        root = Path(".")
        my_file = Path(root/"explainer_outputs"/"LIME_pickles"/(model_name+'_LIME_SP_'+self.grid))
        
        if my_file.is_file():
            print("EXISTS!!!!!!!!!!!!!!")
            pickle_in = open(my_file,"rb")
            sp_obj = pickle.load(pickle_in)
            print("LOADED!!!!!!!!!!!!!!")
        else:
            print("DOESNT EXIST. CREATING NEW")
            start = timer()
            sp_obj = submodular_pick.SubmodularPick(explainer, np.array(self.X_train), predict_fn, num_features=self.X_test.shape[1], num_exps_desired=5)
            end = timer()
            print('Global LIME Explanations: ', end-start)
            
            
            #Store in Pickle
            pickle_out = open(my_file,"wb")
            pickle.dump(sp_obj, pickle_out)
            pickle_out.close()
            
            
            
            spExplanationTime = end-start
            print('LIME Global Explanation time: ',spExplanationTime)
        
        dir_name = os.path.join('explainer_outputs','LIME','Global',self.grid)
        pathGlobal = os.path.join(dir_name,model_name + 'LIME_SP.pdf')
        with PdfPages(pathGlobal) as pdf:
            for exp in sp_obj.sp_explanations:
                fig = exp.as_pyplot_figure(label=exp.available_labels()[0])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        return sp_obj
    
    
    def generateGlobalShapExplanations(self):
        model_name = type(self.model).__name__
        if(model_name == 'Sequential'):
            if(self.model.name == 'DNN'):
                predict_fn = self.dnn_model_predict
                model_name = 'DNN'
            elif(self.model.name == 'RNN'):
                predict_fn = self.rnn_model_predict
                model_name = 'RNN'
        else:
            predict_fn = self.model.predict_proba
        dir_name = os.path.join('explainer_outputs','SHAP',self.grid)
        
        file_name = model_name + "SHAP.pdf"
        pathGlobal = os.path.join(dir_name, file_name)

        #Check if SHAP_Values of the specific classifier is already stored 
        root = Path(".")
        my_file = Path(root/"explainer_outputs"/"SHAP_pickles"/(model_name+'SHAP'+self.grid))
        
        if my_file.is_file():
            print("EXISTS!!!!!!!!!!!!!!")
            pickle_in = open(my_file,"rb")
            shap_values = pickle.load(pickle_in)
            print("LOADED!!!!!!!!!!!!!!")
        else:
            print("DOESNT EXIST. CREATING NEW")
            data = shap.kmeans(self.X_train, 100)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start = timer()
               
                
                
                explainer = shap.KernelExplainer(predict_fn, data)
                shap_values = explainer.shap_values(self.X_test)
                end = timer()
                print("SHAP Calculation Time: ", end-start )
                #Store SHAP Values
                
                pickle_out = open(my_file,"wb")
                pickle.dump(shap_values, pickle_out)
                pickle_out.close()
            
        with PdfPages(pathGlobal) as pdf:
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False) 
            plt.title(model_name + ' SHAP Global Summary Plot')
            pdf.savefig( bbox_inches='tight')
            plt.close()
            
            shap.summary_plot(shap_values[1], self.X_test, feature_names=self.feature_names, show=False) 
            plt.title(model_name + 'SHAP Global Summary Plot [Predictions for Stability]')
            file_name = model_name + "Global_Summary2_SHAP"
            pdf.savefig( bbox_inches='tight')
            plt.close()
        return shap_values

    def generate_breakdown_explainer(self,i):
        
        
        
        model_name = type(self.model).__name__
        model = self.model
        if(model_name == 'Sequential'):
            X = pd.DataFrame.from_records(self.X_test)

            model_name = self.model.name
            if(self.model.name == 'DNN'):
                instc = self.X_test[i].reshape((1,-1))
                dnn_clf = tf.keras.wrappers.scikit_learn.KerasClassifier(
                                        dnn_model_create,
                                        input=self.X_test.shape[1],
                                        epochs=10,
                                        verbose=False)
                dnn_clf._estimator_type = "classifier"
                dnn_clf.fit(self.X_train, self.y_train)
                model = dnn_clf
            elif(self.model.name == 'RNN'):
                print("")
        else:
            instc = self.X_test.iloc[i]
            X = self.X_test
        
        smartgrid_exp = dx.Explainer(model, X, self.y_test,
                    label = ("Smart Grid New England " + model_name +  " Pipeline on instance: " + str(i)))
        
        X.columns = self.feature_names
        instance = smartgrid_exp.predict_parts(instc, 
                type = 'break_down')
        fig = instance.plot(max_vars=30, show=False)
        
        
        
        fig.write_image("explainer_outputs/Break_Down/"+model_name+"_"+str(i)+"_"+self.grid+".svg")
    

    def generate_surrogate_explainer(self):
        print("Predicting with original Model")
        predictions = self.model.predict(self.X_train)

        logreg = LogisticRegression(solver='lbfgs',max_iter=5000)
        logreg.fit(self.X_train,predictions)

        logrespredictions = logreg.predict(self.X_test)
        complexmodelpredictions = self.model.predict(self.X_test)
        score = r2_score(logrespredictions,complexmodelpredictions)

        print(logreg.score(self.X_test,self.y_test))
        coef = logreg.coef_[0]
        coef = list(map(abs, coef))
        
        sort_index = np.argsort(coef)[::-1]
        sort_features = list(map((lambda x: index_featurer_names[x]),sort_index))
        sort_vals  = np.sort(coef)[::-1]
        

        
        result = dict(zip(sort_features,sort_vals))
        df = pd.DataFrame.from_records([result])
        df = df.T
        
        name =  type(self.model).__name__+"_"+self.grid+".xlsx"
        filepath = 'explainer_outputs/Surrogate/'+name
        df.to_excel(filepath)
        
        print()


    #Get the indexes of the most improtant features
    def get_top_LIME_important_features(self,spobj,m):
        X =spobj.sp_explanations
        exps = []
        for explanation in X:
            explarr = explanation.local_exp.values()
            exps.append(explarr)
            
        flat_flatexps = [item for sublist in exps for item in sublist]
        flat_flatexps = [item for sublist in flat_flatexps for item in sublist]
        
        flat_flatexps = sorted(flat_flatexps, key=lambda x: abs(x[1]), reverse=True)

        d = {x:0 for x, _ in flat_flatexps}

        df = pd.DataFrame.from_records(flat_flatexps, columns =['Index', 'Importance'])
        df = df.abs()
        df = df.groupby(df['Index']).mean(0)

        records = df.to_records(index=True)
        records= sorted(records,key=lambda x: x[1], reverse=True)
        resultlimeindexes = list(records)


    
        # for name, num in flat_flatexps: d[name] += num
    
        # using map
        # output = list(map(tuple, d.items()))

        #important_indexes = [x[0] for x in output ]
        #temp = important_indexes[:m]
        res = [resultlimeindexes][:m]
        
        res = res[0][:m]
        print()
        return res
        


    def get_top_SHAP_important_features(self,shap,m):
        
        feats = shap[1]
        # feats = np.sum(feats, axis=0)

        feats2 = np.abs(feats).mean(0)
        
        
        feat_vals  = np.sort(abs(feats2))[::-1]
        feat_index = np.argsort(-(abs(feats2)))
        
        result = list(zip(feat_index,feat_vals))
        
        result =  [result][0][:m]
        
        

        return result
    
    def get_top_important_features_SHAP(self,shap,m):
        

        indexes_shap = self.get_top_SHAP_important_features(shap,m)
        
        
        important_indexes_shap = [element[0] for element in indexes_shap]
        
        return important_indexes_shap

    
    def get_top_important_features_LIME(self,spobj,m):
        indexes_lime = self.get_top_LIME_important_features(spobj,m)

        
        important_indexes_lime = [element[0] for element in indexes_lime]
        
        
        return important_indexes_lime

 
    def predict_with_modified_features(self,important_columns,important_removed):
        
        
        X_reduced = self.X_test.copy()
        X_reduced.columns = range(self.X_train.shape[1])
        unimportant_columns = [item for item in X_reduced if item not in important_columns]
        
        
        
        if(important_removed):
            #Set Important Columns to 0
            #print('SETTING IMPORTANT COLUMNS TO 0')
            #print(important_columns)
            X_reduced[important_columns] = 0
            
        else:
            #Set unimportant Columns to 0
            #print('SETTING OTHER COLUMNS TO 0')
            #print(important_columns)
            #print(unimportant_columns)
            X_reduced[unimportant_columns] = 0
            print(X_reduced.head(2))
        
        
        
        
        
        predictions = self.model.predict(X_reduced)
        return predictions
  
    def dnn_evaluate_with_modified_features(self,dataset,important_columns,important_removed):
        # Reduce features
            ds = dataset.test
            test_ds = ds.copy()
            
            test_ds.columns = range(test_ds.shape[1])
            important_columns = index
            unimportant_columns = [
                item for item in test_ds if item not in important_columns]
            print()

            # for col in important_columns:
            #       test_ds[col].values[:] = 0
            print("Evaluating with n: ",len(index))
            for col in unimportant_columns[:-1]:
                test_ds[col].values[:] = 0
            
            
            
            test_ds.columns = ['F_%s' % f for f in range(test_ds.shape[1])]
            test_ds.columns = [*test_ds.columns[:-1], 'stable-unstable']
            print(test_ds.head(1))
            test_ds = df_to_dataset(test_ds, shuffle=False,
                            batch_size=config_data['batch_size'])
            
            model.evaluate(test_ds)
            
            self.model.evaluate(test_ds)

    #def rnn_evalute_with_modified_features(self,dataset,important_columns,improtant_removed):





        
def df_to_singlerow(df, shuffle=False, batch_size=32):
    """this function takes any data set in a form
    of pandas data frames (e.g. train, test)
    and convert it to tensors.
    :arg df: a data frame with string columns names
    :arg shuffle: 'Default=True'
    :arg batch_size: 'Default=32'
    :returns feature and labels batches as 'tensors'
    """
    df = df.copy()
    
   
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    dataset = ds.batch(batch_size)
    
    return dataset
 

def dnn_model_create(input):
    tf.keras.backend.clear_session()
    model= tf.keras.Sequential([
                       tf.keras.layers.Dense(10,activation='relu',input_shape=[input]),
                       tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model.compile(
                optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model