import matplotlib.pyplot as plt
import argparse
import pickle
import os
#-----------------------------------------------------------------------------------------------#
# Arg parser
arg = argparse.ArgumentParser()
arg.add_argument('-data', '--data', required=True, help='Path to history data')
args = vars(arg.parse_args())
#-----------------------------------------------------------------------------------------------#
# Get all the history data
print('Importing history data...')
history_names = os.listdir(str(args['data']))

history_dict = {}
for file in history_names:
    pickle_file = open(os.path.join(str(args['data']), file), 'rb')
    history_data = pickle.load(pickle_file)
    pickle_file.close()
    history_dict[file] = history_data
#-----------------------------------------------------------------------------------------------#
# Plot history data
def plots(data):
    acc = data['acc']
    val_acc = data['val_acc']
    n = range(len(acc))
    return acc, val_acc, n

keys_ = list(history_dict.keys())
fig = plt.figure()

# Accuracy

# Subplot 1
plt.subplot(3, 3, 1)
data_1 = history_dict[keys_[0]]
acc_1, val_acc_1, n1 = plots(data_1)
plt.plot(n1, val_acc_1, 'r.')
plt.plot(n1, acc_1, 'b.')
plt.grid()
plt.title(keys_[0])
plt.xlabel('iters')
plt.ylabel('acc')

# Subplot 2
plt.subplot(3, 3, 2)
data_2 = history_dict[keys_[1]]
acc_2, val_acc_2, n2 = plots(data_2)
plt.plot(n2, val_acc_2, 'r.')
plt.plot(n2, acc_2, 'b.')
plt.grid()
plt.title(keys_[1])
plt.xlabel('iters')
plt.ylabel('acc')

# Subplot 3
plt.subplot(3, 3, 3)
data_3 = history_dict[keys_[2]]
acc_3, val_acc_3, n3 = plots(data_3)
plt.plot(n3, val_acc_3, 'r.')
plt.plot(n3, acc_3, 'b.')
plt.grid()
plt.title(keys_[2])
plt.xlabel('iters')
plt.ylabel('acc')

# Subplot 4
plt.subplot(3, 3, 4)
data_4 = history_dict[keys_[3]]
acc_4, val_acc_4, n4 = plots(data_4)
plt.plot(n4, val_acc_4, 'r.')
plt.plot(n4, acc_4, 'b.')
plt.grid()
plt.title(keys_[3])
plt.xlabel('iters')
plt.ylabel('acc')

# Subplot 5
plt.subplot(3, 3, 5)
data_5 = history_dict[keys_[4]]
acc_5, val_acc_5, n5 = plots(data_5)
plt.plot(n5, val_acc_5, 'r.')
plt.plot(n5, acc_5, 'b.')
plt.grid()
plt.title(keys_[4])
plt.xlabel('iters')
plt.ylabel('acc')

# Subplot 6
plt.subplot(3, 3, 6)
data_6 = history_dict[keys_[5]]
acc_6, val_acc_6, n6 = plots(data_6)
plt.plot(n6, val_acc_6, 'r.')
plt.plot(n6, acc_6, 'b.')
plt.grid()
plt.title(keys_[5])
plt.xlabel('iters')
plt.ylabel('acc')

# Subplot 7
plt.subplot(3, 3, 7)
data_7 = history_dict[keys_[6]]
acc_7, val_acc_7, n7 = plots(data_7)
plt.plot(n7, val_acc_7, 'r.')
plt.plot(n7, acc_7, 'b.')
plt.grid()
plt.title(keys_[6])
plt.xlabel('iters')
plt.ylabel('acc')

# Show all plots
plt.show()