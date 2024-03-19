
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10], load_from_file=False)

net.SGD(training_data, 30, 10, 3.0, write_to_file=True, test_data=test_data)
