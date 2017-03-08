package saivenky.neural.image;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

/**
 * Created by saivenky on 1/28/17.
 */
public class ImageWriter {
    public static void write(File f, int rows, int columns, float[] pixels) {
        BufferedImage image = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < columns; c++) {
                int index = r * columns + c;
                float pixel = pixels[index];
                byte val = DataUtils.toPixelByte(pixel);
                image.setRGB(c, r, val);
            }
        }

        try {
            ImageIO.write(image, "png", f);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        File f = new File("/home/saivenky/image-test.png");
        float[] pixels = {0, 0, 1, 0, 1, 0, 1, 0, 0};
        write(f, 3, 3, pixels);
    }
}
