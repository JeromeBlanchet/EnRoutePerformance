import pickle
import os
from GetMIT_Map import MappingMIT

MIT_Enroute_geo = pickle.load(open(os.getcwd() + '/TMI/MIT_WithGeometry_Enroute.p'))
def MainFunction(Dep, Arr):
    print Dep, Arr
    Dep_Arr = MappingMIT(Dep,Arr,2013,MIT_Enroute_geo, Type = 'Nominal')
    Mapping_result = Dep_Arr.Main(parameters = {'AIRWAY':[0.25, 1], 'NAS': 0.25})
    print(Dep_Arr.Count_Max_MIT(Mapping_result))
    MNL_Final = Dep_Arr.MergeWithMNL(Mapping_result)
    MNL_Final.to_csv(os.getcwd() + '/MNL/Final_MNL_' + Dep + Arr + '_2013.csv', index=False)
    pickle.dump(Mapping_result, open(os.getcwd() + '/TMI/MapResult/' + Dep+'_'+Arr+'.p','w'))

MainFunction('BOS','IAH')
