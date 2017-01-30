package saivenky.neural.image;

/**
 * Created by saivenky on 1/28/17.
 */
public class DataUtils {
    public static double toPixel(byte b) {
        int unsigned = (int)b & 0xff;
        return toPixelFromByte(unsigned);
    }

    public static double toPixelFromByte(int byteValue) {
        return normalize(byteValue, 0, 255);
    }

    public static byte toPixelByte(double doubleValue) {
        return (byte)(doubleValue * (Byte.MAX_VALUE - Byte.MIN_VALUE) + Byte.MIN_VALUE);
    }

    public static double normalize(double d, double min, double max) {
        return (d - min) / (max - min);
    }

    public static void main(String[] args) {
        System.out.println(toPixel((byte)-43));
        System.out.println(toPixel((byte)0));
        System.out.println(toPixel((byte)120));
    }
}
