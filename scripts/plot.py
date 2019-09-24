import pickle as pickle
from params import *
import pylab as plt

times = np.linspace(0, t_max, N_t)

#load results
t_min, Val, Test, val_prior, test_prior = pickle.load(open('results/GDR_'+str(dataset)+'_'+ str(tpe_prior)+'.pkl','rb'))

#plot results
plt.figure(figsize=(5,4))
plt.plot(times[:n_min], Val, label='Validation set', lw=5)
plt.plot(times[:n_min], Test, label = 'Test set', lw=2)
plt.axvline(times[t_min],c='k', label='Max of validation')
plt.xlabel(r'$t_{min}$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
#plt.title('Max of validation:'+ str(np.round(np.max(Val),3))+', Max of test:'+ str( np.round(np.max(Test),3)) + ', Test set on max validation:'+ str(np.round(Test[t_min],3)) )
plt.savefig('figures/scan_t_min_' + dataset + '_' + tpe_prior + '.eps', bbox_inches='tight')

print('Prior distribution accuracies:')
print('Validation accuracy score: ' + str(val_prior))
print('Testing accuracy score: ' + str(test_prior))

print('Full accuracies:')
print('Max of validation:', np.max(Val))
print('Max of test:', np.max(Test))
print('Test set on max validation:', Test[t_min])

plt.show()
