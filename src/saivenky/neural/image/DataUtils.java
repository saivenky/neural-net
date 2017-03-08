package saivenky.neural.image;

/**
 * Created by saivenky on 1/28/17.
 */
public class DataUtils {
    public static float toPixel(byte b) {
        int unsigned = (int)b & 0xff;
        return toPixelFromByte(unsigned);
    }

    private static float toPixelFromByte(int byteValue) {
        return normalize(byteValue, 0, 255);
    }

    static byte toPixelByte(float floatValue) {
        return (byte)(floatValue * (Byte.MAX_VALUE - Byte.MIN_VALUE) + Byte.MIN_VALUE);
    }

    private static float normalize(float d, float min, float max) {
        return (d - min) / (max - min);
    }

    public static void main(String[] args) {
        System.out.println(toPixel((byte)-43));
        System.out.println(toPixel((byte)0));
        System.out.println(toPixel((byte)120));
    }
}
