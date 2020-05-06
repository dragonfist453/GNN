IMPORT GNN.Image;
#option('outputLimit', 2000);

//Take MNIST dataset using Image module from suitable logical files
//MNIST train images
mnist_train_images := Image.MNIST_train_images('~test::MNIST_train_images');
OUTPUT(mnist_train_images);

//MNIST train labels
mnist_train_labels := Image.MNIST_train_labels('~test::MNIST_train_labels');
OUTPUT(mnist_train_labels);

//MNIST test images
mnist_test_images := Image.MNIST_test_images('~test::MNIST_test_images');
OUTPUT(mnist_test_images);

//MNIST test labels
mnist_test_labels := Image.MNIST_test_labels('~test::MNIST_test_labels');
OUTPUT(mnist_test_labels);