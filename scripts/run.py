import GDR as gdr
import pickle as pickle
from params import *
import pylab as plt


gdr_obj = gdr.GDR(dataset, tpe_rw)
gdr_obj.load_dataset()

gdr_obj.define_comm_feature()

gdr_obj.define_similarity_distribution(tpe = tpe_prior)


print('Compute the graph Laplacian')
gdr_obj.Laplacian()

#to generate the prior distribution, can be smoothed by paramter tau
tau = 0 
gdr_obj.precondition_laplacian(tpe = 'weighted')
val_acc, test_acc = gdr_obj.precondition_distribution(tau, disp = False)

gdr_obj.t_max = t_max #maximum time for matrix exponentials
gdr_obj.t_min = 0.0 #minimum time for matrix exponentials (it will search for other best larger min times)
gdr_obj.N_t = N_t #number of timesteps

print('Compute the exponentials')
gdr_obj.apply_exponential(disp=True)

print('Scan t_min')
t_min, Val, Test, val_prior, test_prior =  gdr_obj.scan_tmin(n_min, output_full=True)

#save results
pickle.dump([t_min, Val, Test, val_prior, test_prior], open('results/GDR_'+str(dataset)+'_'+ str(tpe_prior)+'.pkl','wb'))

#plot results
plt.figure()
plt.plot(gdr_obj.times[:n_min], Val, label='Validation set')
plt.plot(gdr_obj.times[:n_min], Test, label = 'Test set')
plt.axvline(gdr_obj.times[t_min])
plt.xlabel(r'$t_{min}$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Max of validation:'+ str(np.round(np.max(Val),3))+', Max of test:'+ str( np.round(np.max(Test),3)) + ', Test set on max validation:'+ str(np.round(Test[t_min],3)) )

plt.savefig('images/GDR_'+str(dataset)+'_'+ str(tpe_prior)+'_'+str(tpe_rw)+'.png')
plt.show()

print('Prior distribution before random-walk:')
print('Validation accuracy score: ' + str(val_prior))
print('Testing accuracy score: ' + str(test_prior))

print('After random-walk:')
print('Max of validation:', np.max(Val))
print('Max of test:', np.max(Test))
print('Test set on max validation:', Test[t_min])

