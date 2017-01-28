import java.util.ArrayList;
import java.util.List;

/**
 * Created by saivenky on 1/24/17.
 */
public class Tokenizer {
    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String[] split = text.split("[ ,.:;()\\[\\]{}]+");
        for(String possibleToken : split) {
            tokenizeWord(tokens, possibleToken);
        }

        return tokens;
    }

    private static void tokenizeWord(List<String> tokens, String possibleToken) {
        if (possibleToken.contains("'re")) {
            String[] split = possibleToken.split("'");
            if (split.length == 2) {
                tokens.add(split[0]);
                tokens.add("are");
            }
        }
        else if (possibleToken.equals("I'm")) {
            tokens.add("I");
            tokens.add("am");
        }
        else {
            tokens.add(possibleToken);
        }
    }

    public static void main(String[] args) {
        String text = "Hello, world! I'm a tokenizer. You're a person who will supply me text to tokenize.";
        System.out.println(String.join(" | ", tokenize(text)));
    }
}
