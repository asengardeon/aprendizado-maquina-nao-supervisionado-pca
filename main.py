from pca_functions import load_dataset, model

directory="ORL2"

def print_datasets(train_data, test_data):
    print("############### dataset de treino ##############")
    for p in train_data:
        print(p.id, " - ", p.label)
    print("############### dataset de teste ##############")
    for p in test_data:
        print(p.id, " - ", p.label)



def execute():
    train_data, test_data = load_dataset(directory)
    print_datasets(train_data, test_data)

    component_numbers = list()
    accuracies = list()

    for pca in range(1, 21):
        prediction, accuracy = model(pca, train_data, test_data)
        component_numbers.append(prediction)
        accuracies.append(accuracy)


    for i in range(len(component_numbers)):
        if (i >= 9):
            print('{} componentes principais, acurácia: {}%.'.format(component_numbers[i], accuracies[i]))



if __name__ == '__main__':
    execute()
