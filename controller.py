import oct_classification
import divide_dataset


img_path = 'Data/OCT2019'
test_path = img_path + '/test'
train_path = img_path + '/train'
train_num = 280

'''
Det er blot navne til generering af faktiske filer, modeller og resultater, 
saa naar de ikke eksisterer laves der nogle nye og findes de, saa slettes de gamle
Med forbehold for at jeg ikke har fanget det helt 100%
'''
file_name = 'middle_feature.h5'
model_file = 'model.h5'
result_path = 'result.csv'

layer_num = 150


# Hvorfor 10 gange iteration?
iteration_times = 10

# proposed method
for i in range(iteration_times):
	random_copyfile(img_path, train_path, test_path, train_num)
	extract_feature(train_path, test_path, file_name)
	build_model(file_name, weights_file)
	model_predict(file_name, model_file, result_path)


# fine-tuning method
for i in range(iteration_times):
	random_copyfile(img_path, train_path, test_path, train_num)
	fine_tuning(train_path, layer_num)
	fine_tuning_predict(test_path, model_file, result_path)