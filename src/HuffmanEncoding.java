import java.util.*;

/**
 * Created by saivenky on 1/24/17.
 */
public class HuffmanEncoding {
    public static class Node<TValue extends Comparable<TValue>, TFrequency extends Integer> implements Comparable<Node<TValue, TFrequency>> {
        Node<TValue, TFrequency> left;
        Node<TValue, TFrequency> right;
        Node<TValue, TFrequency> parent;

        TValue value;
        TFrequency frequency;
        String encoding;


        public Node(TValue value, TFrequency frequency) {
            left = null;
            right = null;
            parent = null;
            this.value = value;
            this.frequency = frequency;
        }

        public static <TValue extends Comparable<TValue>, TFrequency extends Integer> Node<TValue, TFrequency> merge(
                Node<TValue, TFrequency> a, Node<TValue, TFrequency> b) {
            Node<TValue, TFrequency> parent = new Node(
                    null, a.frequency.intValue() + b.frequency.intValue());
            a.parent = parent;
            b.parent = parent;
            int compare = a.compareTo(b);
            if (compare < 0) {
                parent.left = a;
                parent.right = b;
            }
            else {
                parent.left = b;
                parent.right = a;
            }

            return parent;
        }

        @Override
        public String toString() {
            if (left == null && right == null) {
                String result = String.format("%s:%s (%s)", value, frequency, encoding);
                return result;
            }
            String result = String.format("%s:%s [ %s | %s ]", value, frequency, left, right);
            return result;
        }

        @Override
        public int compareTo(Node<TValue, TFrequency> o) {
            int compare = frequency.compareTo(o.frequency);
            if (compare == 0) {
                if (value == null) {
                    compare = -1;
                }
                else if (o.value == null) {
                    compare = 1;
                }
                else compare = -value.compareTo(o.value);
            }

            return compare;
        }
    }

    public static TreeSet<Node<Character,Integer>> frequency(String text) {
        HashMap<Character, Integer> frequency = new HashMap<>();
        for(int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            int count = frequency.getOrDefault(c, 0);
            frequency.put(c, count + 1);
        }

        Set<Character> chars = frequency.keySet();
        TreeSet<Node<Character,Integer>> nodes = new TreeSet<>();
        for (char c : chars) {
            nodes.add(new Node<>(c, frequency.get(c)));
        }

        return nodes;
    }

    public static String repeat(String s, int times) {
        StringBuilder result = new StringBuilder();
        for(int i = 0; i < times; i++) {
            result.append(s);
        }

        return result.toString();
    }

    public static void main(String[] args) {
        TreeSet<Node<Character,Integer>> nodes = frequency(
                "j'aime aller sur le bord de l'eau les jeudis ou les jours impairs");

        while(nodes.size() > 1) {
            Node smallest = nodes.pollFirst();
            Node secondSmallest = nodes.pollFirst();
            nodes.add(Node.merge(smallest, secondSmallest));
        }

        TreeMap<Character, String> encodings = new TreeMap<>();
        createEncoding(encodings, nodes.first(), "");
        Set<Character> chars = encodings.keySet();
        for(Character c : chars) {
            System.out.printf("%s: %s\n", c, encodings.get(c));
        }
    }

    private static void createEncoding(
            Map<Character, String> encoding, Node<Character, Integer> node, String prefix) {
        node.encoding = prefix;
        if(node.value != null) encoding.put(node.value, node.encoding);
        if (node.left != null) createEncoding(encoding, node.left, prefix + "0");
        if (node.right != null) createEncoding(encoding, node.right, prefix + "1");
    }
}
