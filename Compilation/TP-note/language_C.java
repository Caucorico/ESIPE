// Generated from language_C.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class language_C extends Lexer {
	static { RuntimeMetaData.checkVersion("4.7.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		WHITE_SPACE=1, END_LINE=2, START_BLOCK=3, END_BLOCK=4, START_CONTENT=5, 
		END_CONTENT=6, CONTENT_SEPARATOR=7, BINARY=8, INT=9, LONG=10, FLOAT=11, 
		DOUBLE=12, INCREMENT=13, DECREMENT=14, AFFECTATION=15, EQUAL=16, INFERIOR=17, 
		SUPERIOR=18, STRING=19, CHAR=20, TYPES=21, INT_TYPE=22, CHAR_TYPE=23, 
		UNSIGNED_TYPE=24, POINTER_TYPE=25, FLOAT_TYPE=26, DOUBLE_TYPE=27, IDENTIFIER=28;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"WHITE_SPACE", "END_LINE", "START_BLOCK", "END_BLOCK", "START_CONTENT", 
			"END_CONTENT", "CONTENT_SEPARATOR", "BINARY", "INT", "LONG", "FLOAT", 
			"DOUBLE", "INCREMENT", "DECREMENT", "AFFECTATION", "EQUAL", "INFERIOR", 
			"SUPERIOR", "STRING", "CHAR", "TYPES", "INT_TYPE", "CHAR_TYPE", "UNSIGNED_TYPE", 
			"POINTER_TYPE", "FLOAT_TYPE", "DOUBLE_TYPE", "IDENTIFIER"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, null, "';'", "'{'", "'}'", "'('", "')'", "','", null, null, null, 
			null, null, "'++'", "'--'", "'='", "'=='", "'<'", "'>'", null, null, 
			null, "'int'", "'char'", "'unsigned'", null, "'float'", "'double'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "WHITE_SPACE", "END_LINE", "START_BLOCK", "END_BLOCK", "START_CONTENT", 
			"END_CONTENT", "CONTENT_SEPARATOR", "BINARY", "INT", "LONG", "FLOAT", 
			"DOUBLE", "INCREMENT", "DECREMENT", "AFFECTATION", "EQUAL", "INFERIOR", 
			"SUPERIOR", "STRING", "CHAR", "TYPES", "INT_TYPE", "CHAR_TYPE", "UNSIGNED_TYPE", 
			"POINTER_TYPE", "FLOAT_TYPE", "DOUBLE_TYPE", "IDENTIFIER"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public language_C(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "language_C.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	@Override
	public void action(RuleContext _localctx, int ruleIndex, int actionIndex) {
		switch (ruleIndex) {
		case 1:
			END_LINE_action((RuleContext)_localctx, actionIndex);
			break;
		case 2:
			START_BLOCK_action((RuleContext)_localctx, actionIndex);
			break;
		case 3:
			END_BLOCK_action((RuleContext)_localctx, actionIndex);
			break;
		case 4:
			START_CONTENT_action((RuleContext)_localctx, actionIndex);
			break;
		case 5:
			END_CONTENT_action((RuleContext)_localctx, actionIndex);
			break;
		case 6:
			CONTENT_SEPARATOR_action((RuleContext)_localctx, actionIndex);
			break;
		case 7:
			BINARY_action((RuleContext)_localctx, actionIndex);
			break;
		case 8:
			INT_action((RuleContext)_localctx, actionIndex);
			break;
		case 9:
			LONG_action((RuleContext)_localctx, actionIndex);
			break;
		case 10:
			FLOAT_action((RuleContext)_localctx, actionIndex);
			break;
		case 11:
			DOUBLE_action((RuleContext)_localctx, actionIndex);
			break;
		case 12:
			INCREMENT_action((RuleContext)_localctx, actionIndex);
			break;
		case 13:
			DECREMENT_action((RuleContext)_localctx, actionIndex);
			break;
		case 14:
			AFFECTATION_action((RuleContext)_localctx, actionIndex);
			break;
		case 15:
			EQUAL_action((RuleContext)_localctx, actionIndex);
			break;
		case 16:
			INFERIOR_action((RuleContext)_localctx, actionIndex);
			break;
		case 17:
			SUPERIOR_action((RuleContext)_localctx, actionIndex);
			break;
		case 18:
			STRING_action((RuleContext)_localctx, actionIndex);
			break;
		case 19:
			CHAR_action((RuleContext)_localctx, actionIndex);
			break;
		case 20:
			TYPES_action((RuleContext)_localctx, actionIndex);
			break;
		case 21:
			INT_TYPE_action((RuleContext)_localctx, actionIndex);
			break;
		case 22:
			CHAR_TYPE_action((RuleContext)_localctx, actionIndex);
			break;
		case 23:
			UNSIGNED_TYPE_action((RuleContext)_localctx, actionIndex);
			break;
		case 24:
			POINTER_TYPE_action((RuleContext)_localctx, actionIndex);
			break;
		case 25:
			FLOAT_TYPE_action((RuleContext)_localctx, actionIndex);
			break;
		case 26:
			DOUBLE_TYPE_action((RuleContext)_localctx, actionIndex);
			break;
		case 27:
			IDENTIFIER_action((RuleContext)_localctx, actionIndex);
			break;
		}
	}
	private void END_LINE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 0:
			 System.out.println("END_LINE"); 
			break;
		}
	}
	private void START_BLOCK_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 1:
			 System.out.println("START BLOCK"); 
			break;
		}
	}
	private void END_BLOCK_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 2:
			 System.out.println("END BLOCK"); 
			break;
		}
	}
	private void START_CONTENT_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 3:
			 System.out.println("START CONTENT"); 
			break;
		}
	}
	private void END_CONTENT_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 4:
			 System.out.println("END CONTENT"); 
			break;
		}
	}
	private void CONTENT_SEPARATOR_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 5:
			 System.out.println("CONTENT SEPARATOR"); 
			break;
		}
	}
	private void BINARY_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 6:
			 System.out.println("Binary"); 
			break;
		}
	}
	private void INT_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 7:
			 System.out.println("Integer"); 
			break;
		}
	}
	private void LONG_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 8:
			 System.out.println("Long"); 
			break;
		}
	}
	private void FLOAT_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 9:
			 System.out.println("Float"); 
			break;
		}
	}
	private void DOUBLE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 10:
			 System.out.println("Double"); 
			break;
		}
	}
	private void INCREMENT_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 11:
			 System.out.println("Incrementation"); 
			break;
		}
	}
	private void DECREMENT_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 12:
			 System.out.println("Decrementation"); 
			break;
		}
	}
	private void AFFECTATION_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 13:
			 System.out.println("Affectation"); 
			break;
		}
	}
	private void EQUAL_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 14:
			 System.out.println("Equal"); 
			break;
		}
	}
	private void INFERIOR_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 15:
			System.out.println("Inferieur"); 
			break;
		}
	}
	private void SUPERIOR_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 16:
			 System.out.println("Superieur"); 
			break;
		}
	}
	private void STRING_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 17:
			 System.out.println("Chaine de caractere"); 
			break;
		}
	}
	private void CHAR_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 18:
			 System.out.println("Caractere"); 
			break;
		}
	}
	private void TYPES_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 19:
			 System.out.println("Type"); 
			break;
		}
	}
	private void INT_TYPE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 20:
			 System.out.println("Integer Type"); 
			break;
		}
	}
	private void CHAR_TYPE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 21:
			 System.out.println("Char Type"); 
			break;
		}
	}
	private void UNSIGNED_TYPE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 22:
			 System.out.println("Unsigned Type"); 
			break;
		}
	}
	private void POINTER_TYPE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 23:
			 System.out.println("Pointer"); 
			break;
		}
	}
	private void FLOAT_TYPE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 24:
			 System.out.println("Float"); 
			break;
		}
	}
	private void DOUBLE_TYPE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 25:
			 System.out.println("Double"); 
			break;
		}
	}
	private void IDENTIFIER_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 26:
			 System.out.println("Identificateur"); 
			break;
		}
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\36\u00fb\b\1\4\2"+
		"\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4"+
		"\13\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22"+
		"\t\22\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31"+
		"\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\3\2\6\2=\n\2\r\2\16\2>\3"+
		"\2\3\2\3\3\3\3\3\3\3\4\3\4\3\4\3\5\3\5\3\5\3\6\3\6\3\6\3\7\3\7\3\7\3\b"+
		"\3\b\3\b\3\t\3\t\3\t\3\t\6\tY\n\t\r\t\16\tZ\3\t\3\t\3\n\6\n`\n\n\r\n\16"+
		"\na\3\n\3\n\3\13\6\13g\n\13\r\13\16\13h\3\13\3\13\3\13\3\f\6\fo\n\f\r"+
		"\f\16\fp\3\f\5\ft\n\f\3\f\7\fw\n\f\f\f\16\fz\13\f\3\f\3\f\3\f\3\r\6\r"+
		"\u0080\n\r\r\r\16\r\u0081\3\r\3\r\6\r\u0086\n\r\r\r\16\r\u0087\3\r\3\r"+
		"\3\16\3\16\3\16\3\16\3\16\3\17\3\17\3\17\3\17\3\17\3\20\3\20\3\20\3\21"+
		"\3\21\3\21\3\21\3\21\3\22\3\22\3\22\3\23\3\23\3\23\3\24\3\24\7\24\u00a6"+
		"\n\24\f\24\16\24\u00a9\13\24\3\24\3\24\3\24\3\25\3\25\5\25\u00b0\n\25"+
		"\3\25\3\25\3\25\3\26\3\26\3\26\3\26\3\26\3\26\5\26\u00bb\n\26\3\26\3\26"+
		"\3\27\3\27\3\27\3\27\3\27\3\27\3\30\3\30\3\30\3\30\3\30\3\30\3\30\3\31"+
		"\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\32\6\32\u00d8\n\32"+
		"\r\32\16\32\u00d9\3\32\3\32\3\33\3\33\3\33\3\33\3\33\3\33\3\33\3\33\3"+
		"\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\35\6\35\u00f0\n\35\r\35"+
		"\16\35\u00f1\3\35\7\35\u00f5\n\35\f\35\16\35\u00f8\13\35\3\35\3\35\4\u00a7"+
		"\u00af\2\36\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27\r\31\16\33"+
		"\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33\65\34\67"+
		"\359\36\3\2\13\5\2\13\f\17\17\"\"\3\2\62\63\3\2\62;\4\2NNnn\3\2\60\60"+
		"\4\2HHhh\3\2,,\4\2C\\c|\5\2\62;C\\c|\2\u010d\2\3\3\2\2\2\2\5\3\2\2\2\2"+
		"\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2"+
		"\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2"+
		"\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2"+
		"\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2\2"+
		"\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\3<\3\2\2\2\5B\3\2\2\2\7E\3\2\2"+
		"\2\tH\3\2\2\2\13K\3\2\2\2\rN\3\2\2\2\17Q\3\2\2\2\21T\3\2\2\2\23_\3\2\2"+
		"\2\25f\3\2\2\2\27n\3\2\2\2\31\177\3\2\2\2\33\u008b\3\2\2\2\35\u0090\3"+
		"\2\2\2\37\u0095\3\2\2\2!\u0098\3\2\2\2#\u009d\3\2\2\2%\u00a0\3\2\2\2\'"+
		"\u00a3\3\2\2\2)\u00ad\3\2\2\2+\u00ba\3\2\2\2-\u00be\3\2\2\2/\u00c4\3\2"+
		"\2\2\61\u00cb\3\2\2\2\63\u00d7\3\2\2\2\65\u00dd\3\2\2\2\67\u00e5\3\2\2"+
		"\29\u00ef\3\2\2\2;=\t\2\2\2<;\3\2\2\2=>\3\2\2\2><\3\2\2\2>?\3\2\2\2?@"+
		"\3\2\2\2@A\b\2\2\2A\4\3\2\2\2BC\7=\2\2CD\b\3\3\2D\6\3\2\2\2EF\7}\2\2F"+
		"G\b\4\4\2G\b\3\2\2\2HI\7\177\2\2IJ\b\5\5\2J\n\3\2\2\2KL\7*\2\2LM\b\6\6"+
		"\2M\f\3\2\2\2NO\7+\2\2OP\b\7\7\2P\16\3\2\2\2QR\7.\2\2RS\b\b\b\2S\20\3"+
		"\2\2\2TU\7\62\2\2UV\7d\2\2VX\3\2\2\2WY\t\3\2\2XW\3\2\2\2YZ\3\2\2\2ZX\3"+
		"\2\2\2Z[\3\2\2\2[\\\3\2\2\2\\]\b\t\t\2]\22\3\2\2\2^`\t\4\2\2_^\3\2\2\2"+
		"`a\3\2\2\2a_\3\2\2\2ab\3\2\2\2bc\3\2\2\2cd\b\n\n\2d\24\3\2\2\2eg\t\4\2"+
		"\2fe\3\2\2\2gh\3\2\2\2hf\3\2\2\2hi\3\2\2\2ij\3\2\2\2jk\t\5\2\2kl\b\13"+
		"\13\2l\26\3\2\2\2mo\t\4\2\2nm\3\2\2\2op\3\2\2\2pn\3\2\2\2pq\3\2\2\2qs"+
		"\3\2\2\2rt\t\6\2\2sr\3\2\2\2st\3\2\2\2tx\3\2\2\2uw\t\4\2\2vu\3\2\2\2w"+
		"z\3\2\2\2xv\3\2\2\2xy\3\2\2\2y{\3\2\2\2zx\3\2\2\2{|\t\7\2\2|}\b\f\f\2"+
		"}\30\3\2\2\2~\u0080\t\4\2\2\177~\3\2\2\2\u0080\u0081\3\2\2\2\u0081\177"+
		"\3\2\2\2\u0081\u0082\3\2\2\2\u0082\u0083\3\2\2\2\u0083\u0085\7\60\2\2"+
		"\u0084\u0086\t\4\2\2\u0085\u0084\3\2\2\2\u0086\u0087\3\2\2\2\u0087\u0085"+
		"\3\2\2\2\u0087\u0088\3\2\2\2\u0088\u0089\3\2\2\2\u0089\u008a\b\r\r\2\u008a"+
		"\32\3\2\2\2\u008b\u008c\7-\2\2\u008c\u008d\7-\2\2\u008d\u008e\3\2\2\2"+
		"\u008e\u008f\b\16\16\2\u008f\34\3\2\2\2\u0090\u0091\7/\2\2\u0091\u0092"+
		"\7/\2\2\u0092\u0093\3\2\2\2\u0093\u0094\b\17\17\2\u0094\36\3\2\2\2\u0095"+
		"\u0096\7?\2\2\u0096\u0097\b\20\20\2\u0097 \3\2\2\2\u0098\u0099\7?\2\2"+
		"\u0099\u009a\7?\2\2\u009a\u009b\3\2\2\2\u009b\u009c\b\21\21\2\u009c\""+
		"\3\2\2\2\u009d\u009e\7>\2\2\u009e\u009f\b\22\22\2\u009f$\3\2\2\2\u00a0"+
		"\u00a1\7@\2\2\u00a1\u00a2\b\23\23\2\u00a2&\3\2\2\2\u00a3\u00a7\7$\2\2"+
		"\u00a4\u00a6\13\2\2\2\u00a5\u00a4\3\2\2\2\u00a6\u00a9\3\2\2\2\u00a7\u00a8"+
		"\3\2\2\2\u00a7\u00a5\3\2\2\2\u00a8\u00aa\3\2\2\2\u00a9\u00a7\3\2\2\2\u00aa"+
		"\u00ab\7$\2\2\u00ab\u00ac\b\24\24\2\u00ac(\3\2\2\2\u00ad\u00af\7)\2\2"+
		"\u00ae\u00b0\13\2\2\2\u00af\u00b0\3\2\2\2\u00af\u00ae\3\2\2\2\u00b0\u00b1"+
		"\3\2\2\2\u00b1\u00b2\7)\2\2\u00b2\u00b3\b\25\25\2\u00b3*\3\2\2\2\u00b4"+
		"\u00bb\5-\27\2\u00b5\u00bb\5/\30\2\u00b6\u00bb\5\61\31\2\u00b7\u00bb\5"+
		"\63\32\2\u00b8\u00bb\5\65\33\2\u00b9\u00bb\5\67\34\2\u00ba\u00b4\3\2\2"+
		"\2\u00ba\u00b5\3\2\2\2\u00ba\u00b6\3\2\2\2\u00ba\u00b7\3\2\2\2\u00ba\u00b8"+
		"\3\2\2\2\u00ba\u00b9\3\2\2\2\u00bb\u00bc\3\2\2\2\u00bc\u00bd\b\26\26\2"+
		"\u00bd,\3\2\2\2\u00be\u00bf\7k\2\2\u00bf\u00c0\7p\2\2\u00c0\u00c1\7v\2"+
		"\2\u00c1\u00c2\3\2\2\2\u00c2\u00c3\b\27\27\2\u00c3.\3\2\2\2\u00c4\u00c5"+
		"\7e\2\2\u00c5\u00c6\7j\2\2\u00c6\u00c7\7c\2\2\u00c7\u00c8\7t\2\2\u00c8"+
		"\u00c9\3\2\2\2\u00c9\u00ca\b\30\30\2\u00ca\60\3\2\2\2\u00cb\u00cc\7w\2"+
		"\2\u00cc\u00cd\7p\2\2\u00cd\u00ce\7u\2\2\u00ce\u00cf\7k\2\2\u00cf\u00d0"+
		"\7i\2\2\u00d0\u00d1\7p\2\2\u00d1\u00d2\7g\2\2\u00d2\u00d3\7f\2\2\u00d3"+
		"\u00d4\3\2\2\2\u00d4\u00d5\b\31\31\2\u00d5\62\3\2\2\2\u00d6\u00d8\t\b"+
		"\2\2\u00d7\u00d6\3\2\2\2\u00d8\u00d9\3\2\2\2\u00d9\u00d7\3\2\2\2\u00d9"+
		"\u00da\3\2\2\2\u00da\u00db\3\2\2\2\u00db\u00dc\b\32\32\2\u00dc\64\3\2"+
		"\2\2\u00dd\u00de\7h\2\2\u00de\u00df\7n\2\2\u00df\u00e0\7q\2\2\u00e0\u00e1"+
		"\7c\2\2\u00e1\u00e2\7v\2\2\u00e2\u00e3\3\2\2\2\u00e3\u00e4\b\33\33\2\u00e4"+
		"\66\3\2\2\2\u00e5\u00e6\7f\2\2\u00e6\u00e7\7q\2\2\u00e7\u00e8\7w\2\2\u00e8"+
		"\u00e9\7d\2\2\u00e9\u00ea\7n\2\2\u00ea\u00eb\7g\2\2\u00eb\u00ec\3\2\2"+
		"\2\u00ec\u00ed\b\34\34\2\u00ed8\3\2\2\2\u00ee\u00f0\t\t\2\2\u00ef\u00ee"+
		"\3\2\2\2\u00f0\u00f1\3\2\2\2\u00f1\u00ef\3\2\2\2\u00f1\u00f2\3\2\2\2\u00f2"+
		"\u00f6\3\2\2\2\u00f3\u00f5\t\n\2\2\u00f4\u00f3\3\2\2\2\u00f5\u00f8\3\2"+
		"\2\2\u00f6\u00f4\3\2\2\2\u00f6\u00f7\3\2\2\2\u00f7\u00f9\3\2\2\2\u00f8"+
		"\u00f6\3\2\2\2\u00f9\u00fa\b\35\35\2\u00fa:\3\2\2\2\22\2>Zahpsx\u0081"+
		"\u0087\u00a7\u00af\u00ba\u00d9\u00f1\u00f6\36\b\2\2\3\3\2\3\4\3\3\5\4"+
		"\3\6\5\3\7\6\3\b\7\3\t\b\3\n\t\3\13\n\3\f\13\3\r\f\3\16\r\3\17\16\3\20"+
		"\17\3\21\20\3\22\21\3\23\22\3\24\23\3\25\24\3\26\25\3\27\26\3\30\27\3"+
		"\31\30\3\32\31\3\33\32\3\34\33\3\35\34";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}