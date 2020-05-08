IMPORT GNN.Image;

//Input filename of image from landing zone or from a logical file sprayed from landing zone as a blob. Logical file works great for a set of images.
ImageDs := Image.GetImages('~test::images_ds', [100,100]);

//Outputs id, image in data format and dims of the image as modified
//OUTPUT(ImageDs);

//Convert obtained image dataset to tensors
ImageTens := Image.ImgtoTens(ImageDs);

//Output the tensor
//OUTPUT(ImageTens);

//Convert tensor back to image dataset to output
OutputImageDs := Image.TenstoImg(ImageTens);

//Output the image
//OUTPUT(OutputImageDs);

//Output the image to despray as a grid
OutputGrid := Image.OutputGrid(OutputImageDs,2,1, 'test');

//Output grid to despray as png
OUTPUT(OutputGrid, ,'~test::output_image_ds',OVERWRITE);