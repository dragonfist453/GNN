IMPORT GNN.Image;

//Input filename of image from landing zone or from a logical file sprayed from landing zone as a blob. Logical file works great for a set of images.
ImageDs := Image.GetImages('~test::images_ds', [100,100]);

OUTPUT(ImageDs);