import java.awt.image.RescaleOp;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by saivenky on 1/23/17.
 */
public class ShannonCoding {
    public static Map<Character, Integer> frequencyMapping(String text) {
        HashMap<Character, Integer> map = new HashMap<>();
        for(int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            int oldValue = map.getOrDefault(c, 0);
            map.put(c, oldValue + 1);
        }

        return map;
    }

    public static void printMap(Map<Character, Integer> map) {
        Set<Character> chars = map.keySet();
        for(Character c : chars) {
            System.out.printf("%s: %d\n", c, map.get(c));
        }
    }

    public static void binary(double f, double total) {
        total = total / 2;
        while (total > 1e-8) {
            if (f >= total) {
                System.out.print(1);
                f -= total;
            }
            else {
                System.out.print(0);
            }

            total = total / 2;
        }

        System.out.println();
    }

    public static void main(String[] args) {
        String text = "1515151524242424242424242422233";
        Map<Character,Integer> map = frequencyMapping(text);
        printMap(map);

        int i = 0;
        int total = 7;
        while (i++ < 7) binary(i , total);
    }
}
