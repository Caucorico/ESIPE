// Generated from td3_ex1.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class td3_ex1 extends Lexer {
	static { RuntimeMetaData.checkVersion("4.7.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		FRENCH=1, ENGLISH=2, OTHER=3, CLEAR=4, SKIP_IN=5;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"FRENCH", "ENGLISH", "OTHER", "CLEAR", "SKIP_IN"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "FRENCH", "ENGLISH", "OTHER", "CLEAR", "SKIP_IN"
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


	public td3_ex1(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "td3_ex1.g4"; }

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
		case 0:
			FRENCH_action((RuleContext)_localctx, actionIndex);
			break;
		case 1:
			ENGLISH_action((RuleContext)_localctx, actionIndex);
			break;
		}
	}
	private void FRENCH_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 0:
			 System.out.println("French ");  
			break;
		}
	}
	private void ENGLISH_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 1:
			 System.out.println("English ");  
			break;
		}
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\7Z\b\1\4\2\t\2\4"+
		"\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2"+
		"\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2\"\n\2\3\2\3\2\3\3\3\3\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\5\3@\n\3\3\3\3\3\3\4\6\4E\n\4\r\4\16\4F\3\4\3\4"+
		"\3\5\6\5L\n\5\r\5\16\5M\3\5\3\5\3\6\5\6S\n\6\3\6\3\6\3\6\3\6\5\6Y\n\6"+
		"\2\2\7\3\3\5\4\7\5\t\6\13\7\3\2\2\4\u0297\2\2\2B\2]\2b\2}\2\u00ab\2\u00ad"+
		"\2\u00b6\2\u00b8\2\u00bb\2\u00bd\2\u00c1\2\u00d9\2\u00d9\2\u00f9\2\u00f9"+
		"\2\u02c4\2\u02c7\2\u02d4\2\u02e1\2\u02e7\2\u02ed\2\u02ef\2\u02ef\2\u02f1"+
		"\2\u0346\2\u0348\2\u0371\2\u0377\2\u0377\2\u037a\2\u037b\2\u0380\2\u0380"+
		"\2\u0382\2\u0387\2\u0389\2\u0389\2\u038d\2\u038d\2\u038f\2\u038f\2\u03a4"+
		"\2\u03a4\2\u03f8\2\u03f8\2\u0484\2\u048b\2\u0532\2\u0532\2\u0559\2\u055a"+
		"\2\u055c\2\u0562\2\u058a\2\u05b1\2\u05c0\2\u05c0\2\u05c2\2\u05c2\2\u05c5"+
		"\2\u05c5\2\u05c8\2\u05c8\2\u05ca\2\u05d1\2\u05ed\2\u05f1\2\u05f5\2\u0611"+
		"\2\u061d\2\u0621\2\u065a\2\u065a\2\u0662\2\u066f\2\u06d6\2\u06d6\2\u06df"+
		"\2\u06e2\2\u06eb\2\u06ee\2\u06f2\2\u06fb\2\u06ff\2\u0700\2\u0702\2\u0711"+
		"\2\u0742\2\u074e\2\u07b4\2\u07cb\2\u07ed\2\u07f5\2\u07f8\2\u07fb\2\u07fd"+
		"\2\u0801\2\u081a\2\u081b\2\u082f\2\u0841\2\u085b\2\u0861\2\u086d\2\u08a1"+
		"\2\u08b7\2\u08b7\2\u08c0\2\u08d5\2\u08e2\2\u08e4\2\u08ec\2\u08f1\2\u093e"+
		"\2\u093e\2\u094f\2\u094f\2\u0953\2\u0956\2\u0966\2\u0972\2\u0986\2\u0986"+
		"\2\u098f\2\u0990\2\u0993\2\u0994\2\u09ab\2\u09ab\2\u09b3\2\u09b3\2\u09b5"+
		"\2\u09b7\2\u09bc\2\u09be\2\u09c7\2\u09c8\2\u09cb\2\u09cc\2\u09cf\2\u09cf"+
		"\2\u09d1\2\u09d8\2\u09da\2\u09dd\2\u09e0\2\u09e0\2\u09e6\2\u09f1\2\u09f4"+
		"\2\u09fd\2\u09ff\2\u0a02\2\u0a06\2\u0a06\2\u0a0d\2\u0a10\2\u0a13\2\u0a14"+
		"\2\u0a2b\2\u0a2b\2\u0a33\2\u0a33\2\u0a36\2\u0a36\2\u0a39\2\u0a39\2\u0a3c"+
		"\2\u0a3f\2\u0a45\2\u0a48\2\u0a4b\2\u0a4c\2\u0a4f\2\u0a52\2\u0a54\2\u0a5a"+
		"\2\u0a5f\2\u0a5f\2\u0a61\2\u0a71\2\u0a78\2\u0a82\2\u0a86\2\u0a86\2\u0a90"+
		"\2\u0a90\2\u0a94\2\u0a94\2\u0aab\2\u0aab\2\u0ab3\2\u0ab3\2\u0ab6\2\u0ab6"+
		"\2\u0abc\2\u0abe\2\u0ac8\2\u0ac8\2\u0acc\2\u0acc\2\u0acf\2\u0ad1\2\u0ad3"+
		"\2\u0ae1\2\u0ae6\2\u0afa\2\u0aff\2\u0b02\2\u0b06\2\u0b06\2\u0b0f\2\u0b10"+
		"\2\u0b13\2\u0b14\2\u0b2b\2\u0b2b\2\u0b33\2\u0b33\2\u0b36\2\u0b36\2\u0b3c"+
		"\2\u0b3e\2\u0b47\2\u0b48\2\u0b4b\2\u0b4c\2\u0b4f\2\u0b57\2\u0b5a\2\u0b5d"+
		"\2\u0b60\2\u0b60\2\u0b66\2\u0b72\2\u0b74\2\u0b83\2\u0b86\2\u0b86\2\u0b8d"+
		"\2\u0b8f\2\u0b93\2\u0b93\2\u0b98\2\u0b9a\2\u0b9d\2\u0b9d\2\u0b9f\2\u0b9f"+
		"\2\u0ba2\2\u0ba4\2\u0ba7\2\u0ba9\2\u0bad\2\u0baf\2\u0bbc\2\u0bbf\2\u0bc5"+
		"\2\u0bc7\2\u0bcb\2\u0bcb\2\u0bcf\2\u0bd1\2\u0bd3\2\u0bd8\2\u0bda\2\u0c01"+
		"\2\u0c06\2\u0c06\2\u0c0f\2\u0c0f\2\u0c13\2\u0c13\2\u0c2b\2\u0c2b\2\u0c3c"+
		"\2\u0c3e\2\u0c47\2\u0c47\2\u0c4b\2\u0c4b\2\u0c4f\2\u0c56\2\u0c59\2\u0c59"+
		"\2\u0c5d\2\u0c61\2\u0c66\2\u0c81\2\u0c86\2\u0c86\2\u0c8f\2\u0c8f\2\u0c93"+
		"\2\u0c93\2\u0cab\2\u0cab\2\u0cb6\2\u0cb6\2\u0cbc\2\u0cbe\2\u0cc7\2\u0cc7"+
		"\2\u0ccb\2\u0ccb\2\u0ccf\2\u0cd6\2\u0cd9\2\u0cdf\2\u0ce1\2\u0ce1\2\u0ce6"+
		"\2\u0cf2\2\u0cf5\2\u0d01\2\u0d06\2\u0d06\2\u0d0f\2\u0d0f\2\u0d13\2\u0d13"+
		"\2\u0d3d\2\u0d3e\2\u0d47\2\u0d47\2\u0d4b\2\u0d4b\2\u0d4f\2\u0d4f\2\u0d51"+
		"\2\u0d55\2\u0d5a\2\u0d60\2\u0d66\2\u0d7b\2\u0d82\2\u0d83\2\u0d86\2\u0d86"+
		"\2\u0d99\2\u0d9b\2\u0db4\2\u0db4\2\u0dbe\2\u0dbe\2\u0dc0\2\u0dc1\2\u0dc9"+
		"\2\u0dd0\2\u0dd7\2\u0dd7\2\u0dd9\2\u0dd9\2\u0de2\2\u0df3\2\u0df6\2\u0e02"+
		"\2\u0e3d\2\u0e41\2\u0e49\2\u0e4e\2\u0e50\2\u0e82\2\u0e85\2\u0e85\2\u0e87"+
		"\2\u0e88\2\u0e8b\2\u0e8b\2\u0e8d\2\u0e8e\2\u0e90\2\u0e95\2\u0e9a\2\u0e9a"+
		"\2\u0ea2\2\u0ea2\2\u0ea6\2\u0ea6\2\u0ea8\2\u0ea8\2\u0eaa\2\u0eab\2\u0eae"+
		"\2\u0eae\2\u0ebc\2\u0ebc\2\u0ec0\2\u0ec1\2\u0ec7\2\u0ec7\2\u0ec9\2\u0ece"+
		"\2\u0ed0\2\u0edd\2\u0ee2\2\u0f01\2\u0f03\2\u0f41\2\u0f4a\2\u0f4a\2\u0f6f"+
		"\2\u0f72\2\u0f84\2\u0f89\2\u0f9a\2\u0f9a\2\u0fbf\2\u1001\2\u1039\2\u1039"+
		"\2\u103b\2\u103c\2\u1042\2\u1051\2\u1065\2\u1066\2\u106b\2\u106f\2\u1089"+
		"\2\u108f\2\u1091\2\u109d\2\u10a0\2\u10a1\2\u10c8\2\u10c8\2\u10ca\2\u10ce"+
		"\2\u10d0\2\u10d1\2\u10fd\2\u10fd\2\u124b\2\u124b\2\u1250\2\u1251\2\u1259"+
		"\2\u1259\2\u125b\2\u125b\2\u1260\2\u1261\2\u128b\2\u128b\2\u1290\2\u1291"+
		"\2\u12b3\2\u12b3\2\u12b8\2\u12b9\2\u12c1\2\u12c1\2\u12c3\2\u12c3\2\u12c8"+
		"\2\u12c9\2\u12d9\2\u12d9\2\u1313\2\u1313\2\u1318\2\u1319\2\u135d\2\u1360"+
		"\2\u1362\2\u1381\2\u1392\2\u13a1\2\u13f8\2\u13f9\2\u1400\2\u1402\2\u166f"+
		"\2\u1670\2\u1682\2\u1682\2\u169d\2\u16a1\2\u16ed\2\u16ef\2\u16fb\2\u1701"+
		"\2\u170f\2\u170f\2\u1716\2\u1721\2\u1736\2\u1741\2\u1756\2\u1761\2\u176f"+
		"\2\u176f\2\u1773\2\u1773\2\u1776\2\u1781\2\u17b6\2\u17b7\2\u17cb\2\u17d8"+
		"\2\u17da\2\u17dd\2\u17df\2\u1821\2\u187a\2\u1881\2\u18ad\2\u18b1\2\u18f8"+
		"\2\u1901\2\u1921\2\u1921\2\u192e\2\u1931\2\u193b\2\u1951\2\u1970\2\u1971"+
		"\2\u1977\2\u1981\2\u19ae\2\u19b1\2\u19cc\2\u1a01\2\u1a1e\2\u1a21\2\u1a61"+
		"\2\u1a62\2\u1a77\2\u1aa8\2\u1aaa\2\u1b01\2\u1b36\2\u1b36\2\u1b46\2\u1b46"+
		"\2\u1b4e\2\u1b81\2\u1bac\2\u1bad\2\u1bb2\2\u1bbb\2\u1be8\2\u1be8\2\u1bf4"+
		"\2\u1c01\2\u1c38\2\u1c4e\2\u1c52\2\u1c5b\2\u1c80\2\u1c81\2\u1c8b\2\u1cea"+
		"\2\u1cef\2\u1cef\2\u1cf6\2\u1cf6\2\u1cf9\2\u1d01\2\u1dc2\2\u1de8\2\u1df7"+
		"\2\u1e01\2\u1f18\2\u1f19\2\u1f20\2\u1f21\2\u1f48\2\u1f49\2\u1f50\2\u1f51"+
		"\2\u1f5a\2\u1f5a\2\u1f5c\2\u1f5c\2\u1f5e\2\u1f5e\2\u1f60\2\u1f60\2\u1f80"+
		"\2\u1f81\2\u1fb7\2\u1fb7\2\u1fbf\2\u1fbf\2\u1fc1\2\u1fc3\2\u1fc7\2\u1fc7"+
		"\2\u1fcf\2\u1fd1\2\u1fd6\2\u1fd7\2\u1fde\2\u1fe1\2\u1fef\2\u1ff3\2\u1ff7"+
		"\2\u1ff7\2\u1fff\2\u2072\2\u2074\2\u2080\2\u2082\2\u2091\2\u209f\2\u2103"+
		"\2\u2105\2\u2108\2\u210a\2\u210b\2\u2116\2\u2116\2\u2118\2\u211a\2\u2120"+
		"\2\u2125\2\u2127\2\u2127\2\u2129\2\u2129\2\u212b\2\u212b\2\u2130\2\u2130"+
		"\2\u213c\2\u213d\2\u2142\2\u2146\2\u214c\2\u214f\2\u2151\2\u2161\2\u218b"+
		"\2\u24b7\2\u24ec\2\u2c01\2\u2c31\2\u2c31\2\u2c61\2\u2c61\2\u2ce7\2\u2cec"+
		"\2\u2cf1\2\u2cf3\2\u2cf6\2\u2d01\2\u2d28\2\u2d28\2\u2d2a\2\u2d2e\2\u2d30"+
		"\2\u2d31\2\u2d6a\2\u2d70\2\u2d72\2\u2d81\2\u2d99\2\u2da1\2\u2da9\2\u2da9"+
		"\2\u2db1\2\u2db1\2\u2db9\2\u2db9\2\u2dc1\2\u2dc1\2\u2dc9\2\u2dc9\2\u2dd1"+
		"\2\u2dd1\2\u2dd9\2\u2dd9\2\u2de1\2\u2de1\2\u2e02\2\u2e30\2\u2e32\2\u3006"+
		"\2\u300a\2\u3022\2\u302c\2\u3032\2\u3038\2\u3039\2\u303f\2\u3042\2\u3099"+
		"\2\u309e\2\u30a2\2\u30a2\2\u30fd\2\u30fd\2\u3102\2\u3106\2\u3131\2\u3132"+
		"\2\u3191\2\u31a1\2\u31bd\2\u31f1\2\u3202\2\u3401\2\u4db8\2\u4e01\2\u9fed"+
		"\2\ua001\2\ua48f\2\ua4d1\2\ua500\2\ua501\2\ua60f\2\ua611\2\ua622\2\ua62b"+
		"\2\ua62e\2\ua641\2\ua671\2\ua675\2\ua67e\2\ua680\2\ua6f2\2\ua718\2\ua722"+
		"\2\ua723\2\ua78b\2\ua78c\2\ua7b1\2\ua7b1\2\ua7ba\2\ua7f8\2\ua804\2\ua804"+
		"\2\ua808\2\ua808\2\ua80d\2\ua80d\2\ua82a\2\ua841\2\ua876\2\ua881\2\ua8c6"+
		"\2\ua8c6\2\ua8c8\2\ua8f3\2\ua8fa\2\ua8fc\2\ua8fe\2\ua8fe\2\ua900\2\ua90b"+
		"\2\ua92d\2\ua931\2\ua955\2\ua961\2\ua97f\2\ua981\2\ua9b5\2\ua9b5\2\ua9c2"+
		"\2\ua9d0\2\ua9d2\2\ua9e1\2\ua9e7\2\ua9e7\2\ua9f2\2\ua9fb\2\uaa01\2\uaa01"+
		"\2\uaa39\2\uaa41\2\uaa50\2\uaa61\2\uaa79\2\uaa7b\2\uaa7d\2\uaa7f\2\uaac1"+
		"\2\uaac1\2\uaac3\2\uaac3\2\uaac5\2\uaadc\2\uaae0\2\uaae1\2\uaaf2\2\uaaf3"+
		"\2\uaaf8\2\uab02\2\uab09\2\uab0a\2\uab11\2\uab12\2\uab19\2\uab21\2\uab29"+
		"\2\uab29\2\uab31\2\uab31\2\uab5d\2\uab5d\2\uab68\2\uab71\2\uabed\2\uac01"+
		"\2\ud7a6\2\ud7b1\2\ud7c9\2\ud7cc\2\ud7fe\2\uf901\2\ufa70\2\ufa71\2\ufadc"+
		"\2\ufb01\2\ufb09\2\ufb14\2\ufb1a\2\ufb1e\2\ufb2b\2\ufb2b\2\ufb39\2\ufb39"+
		"\2\ufb3f\2\ufb3f\2\ufb41\2\ufb41\2\ufb44\2\ufb44\2\ufb47\2\ufb47\2\ufbb4"+
		"\2\ufbd4\2\ufd40\2\ufd51\2\ufd92\2\ufd93\2\ufdca\2\ufdf1\2\ufdfe\2\ufe71"+
		"\2\ufe77\2\ufe77\2\ufeff\2\uff22\2\uff3d\2\uff42\2\uff5d\2\uff67\2\uffc1"+
		"\2\uffc3\2\uffca\2\uffcb\2\uffd2\2\uffd3\2\uffda\2\uffdb\2\uffdf\2\1\2"+
		"\16\3\16\3)\3)\3=\3=\3@\3@\3P\3Q\3`\3\u0081\3\u00fd\3\u0141\3\u0177\3"+
		"\u0281\3\u029f\3\u02a1\3\u02d3\3\u0301\3\u0322\3\u032e\3\u034d\3\u0351"+
		"\3\u037d\3\u0381\3\u03a0\3\u03a1\3\u03c6\3\u03c9\3\u03d2\3\u03d2\3\u03d8"+
		"\3\u0401\3\u04a0\3\u04b1\3\u04d6\3\u04d9\3\u04fe\3\u0501\3\u052a\3\u0531"+
		"\3\u0566\3\u0601\3\u0739\3\u0741\3\u0758\3\u0761\3\u076a\3\u0801\3\u0808"+
		"\3\u0809\3\u080b\3\u080b\3\u0838\3\u0838\3\u083b\3\u083d\3\u083f\3\u0840"+
		"\3\u0858\3\u0861\3\u0879\3\u0881\3\u08a1\3\u08e1\3\u08f5\3\u08f5\3\u08f8"+
		"\3\u0901\3\u0918\3\u0921\3\u093c\3\u0981\3\u09ba\3\u09bf\3\u09c2\3\u0a01"+
		"\3\u0a06\3\u0a06\3\u0a09\3\u0a0d\3\u0a16\3\u0a16\3\u0a1a\3\u0a1a\3\u0a36"+
		"\3\u0a61\3\u0a7f\3\u0a81\3\u0a9f\3\u0ac1\3\u0aca\3\u0aca\3\u0ae7\3\u0b01"+
		"\3\u0b38\3\u0b41\3\u0b58\3\u0b61\3\u0b75\3\u0b81\3\u0b94\3\u0c01\3\u0c4b"+
		"\3\u0c81\3\u0cb5\3\u0cc1\3\u0cf5\3\u1001\3\u1048\3\u1083\3\u10bb\3\u10d1"+
		"\3\u10eb\3\u1101\3\u1135\3\u1151\3\u1175\3\u1177\3\u1179\3\u1181\3\u11c2"+
		"\3\u11c2\3\u11c7\3\u11db\3\u11dd\3\u11dd\3\u11df\3\u1201\3\u1214\3\u1214"+
		"\3\u1237\3\u1238\3\u123a\3\u123f\3\u1241\3\u1281\3\u1289\3\u1289\3\u128b"+
		"\3\u128b\3\u1290\3\u1290\3\u12a0\3\u12a0\3\u12ab\3\u12b1\3\u12eb\3\u1301"+
		"\3\u1306\3\u1306\3\u130f\3\u1310\3\u1313\3\u1314\3\u132b\3\u132b\3\u1333"+
		"\3\u1333\3\u1336\3\u1336\3\u133c\3\u133e\3\u1347\3\u1348\3\u134b\3\u134c"+
		"\3\u134f\3\u1351\3\u1353\3\u1358\3\u135a\3\u135e\3\u1366\3\u1401\3\u1444"+
		"\3\u1444\3\u1448\3\u1448\3\u144d\3\u1481\3\u14c4\3\u14c5\3\u14c8\3\u14c8"+
		"\3\u14ca\3\u1581\3\u15b8\3\u15b9\3\u15c1\3\u15d9\3\u15e0\3\u1601\3\u1641"+
		"\3\u1641\3\u1643\3\u1645\3\u1647\3\u1681\3\u16b8\3\u1701\3\u171c\3\u171e"+
		"\3\u172d\3\u18a1\3\u18e2\3\u1900\3\u1902\3\u1a01\3\u1a35\3\u1a36\3\u1a41"+
		"\3\u1a51\3\u1a86\3\u1a87\3\u1a9a\3\u1ac1\3\u1afb\3\u1c01\3\u1c0b\3\u1c0b"+
		"\3\u1c39\3\u1c39\3\u1c41\3\u1c41\3\u1c43\3\u1c73\3\u1c92\3\u1c93\3\u1caa"+
		"\3\u1caa\3\u1cb9\3\u1d01\3\u1d09\3\u1d09\3\u1d0c\3\u1d0c\3\u1d39\3\u1d3b"+
		"\3\u1d3d\3\u1d3d\3\u1d40\3\u1d40\3\u1d44\3\u1d44\3\u1d46\3\u1d47\3\u1d4a"+
		"\3\u2001\3\u239c\3\u2401\3\u2471\3\u2481\3\u2546\3\u3001\3\u3431\3\u4401"+
		"\3\u4649\3\u6801\3\u6a3b\3\u6a41\3\u6a61\3\u6ad1\3\u6af0\3\u6b01\3\u6b39"+
		"\3\u6b41\3\u6b46\3\u6b64\3\u6b7a\3\u6b7e\3\u6b92\3\u6f01\3\u6f47\3\u6f51"+
		"\3\u6f81\3\u6f94\3\u6fa2\3\u6fe1\3\u6fe4\3\u7001\3\u87ef\3\u8801\3\u8af5"+
		"\3\ub001\3\ub121\3\ub171\3\ub2fe\3\ubc01\3\ubc6d\3\ubc71\3\ubc7f\3\ubc81"+
		"\3\ubc8b\3\ubc91\3\ubc9c\3\ubc9f\3\ubca1\3\ud401\3\ud457\3\ud457\3\ud49f"+
		"\3\ud49f\3\ud4a2\3\ud4a3\3\ud4a5\3\ud4a6\3\ud4a9\3\ud4aa\3\ud4af\3\ud4af"+
		"\3\ud4bc\3\ud4bc\3\ud4be\3\ud4be\3\ud4c6\3\ud4c6\3\ud508\3\ud508\3\ud50d"+
		"\3\ud50e\3\ud517\3\ud517\3\ud51f\3\ud51f\3\ud53c\3\ud53c\3\ud541\3\ud541"+
		"\3\ud547\3\ud547\3\ud549\3\ud54b\3\ud553\3\ud553\3\ud6a8\3\ud6a9\3\ud6c3"+
		"\3\ud6c3\3\ud6dd\3\ud6dd\3\ud6fd\3\ud6fd\3\ud717\3\ud717\3\ud737\3\ud737"+
		"\3\ud751\3\ud751\3\ud771\3\ud771\3\ud78b\3\ud78b\3\ud7ab\3\ud7ab\3\ud7c5"+
		"\3\ud7c5\3\ud7ce\3\ue001\3\ue009\3\ue009\3\ue01b\3\ue01c\3\ue024\3\ue024"+
		"\3\ue027\3\ue027\3\ue02d\3\ue801\3\ue8c7\3\ue901\3\ue946\3\ue948\3\ue94a"+
		"\3\uee01\3\uee06\3\uee06\3\uee22\3\uee22\3\uee25\3\uee25\3\uee27\3\uee28"+
		"\3\uee2a\3\uee2a\3\uee35\3\uee35\3\uee3a\3\uee3a\3\uee3c\3\uee3c\3\uee3e"+
		"\3\uee43\3\uee45\3\uee48\3\uee4a\3\uee4a\3\uee4c\3\uee4c\3\uee4e\3\uee4e"+
		"\3\uee52\3\uee52\3\uee55\3\uee55\3\uee57\3\uee58\3\uee5a\3\uee5a\3\uee5c"+
		"\3\uee5c\3\uee5e\3\uee5e\3\uee60\3\uee60\3\uee62\3\uee62\3\uee65\3\uee65"+
		"\3\uee67\3\uee68\3\uee6d\3\uee6d\3\uee75\3\uee75\3\uee7a\3\uee7a\3\uee7f"+
		"\3\uee7f\3\uee81\3\uee81\3\uee8c\3\uee8c\3\uee9e\3\ueea2\3\ueea6\3\ueea6"+
		"\3\ueeac\3\ueeac\3\ueebe\3\uf131\3\uf14c\3\uf151\3\uf16c\3\uf171\3\uf18c"+
		"\3\1\3\ua6d9\4\ua701\4\ub737\4\ub741\4\ub820\4\ub821\4\ucea4\4\uceb1\4"+
		"\uebe3\4\uf801\4\ufa20\4\1\22\u0296\2C\2\\\2c\2|\2\u00ac\2\u00ac\2\u00b7"+
		"\2\u00b7\2\u00bc\2\u00bc\2\u00c2\2\u00d8\2\u00da\2\u00f8\2\u00fa\2\u02c3"+
		"\2\u02c8\2\u02d3\2\u02e2\2\u02e6\2\u02ee\2\u02ee\2\u02f0\2\u02f0\2\u0347"+
		"\2\u0347\2\u0372\2\u0376\2\u0378\2\u0379\2\u037c\2\u037f\2\u0381\2\u0381"+
		"\2\u0388\2\u0388\2\u038a\2\u038c\2\u038e\2\u038e\2\u0390\2\u03a3\2\u03a5"+
		"\2\u03f7\2\u03f9\2\u0483\2\u048c\2\u0531\2\u0533\2\u0558\2\u055b\2\u055b"+
		"\2\u0563\2\u0589\2\u05b2\2\u05bf\2\u05c1\2\u05c1\2\u05c3\2\u05c4\2\u05c6"+
		"\2\u05c7\2\u05c9\2\u05c9\2\u05d2\2\u05ec\2\u05f2\2\u05f4\2\u0612\2\u061c"+
		"\2\u0622\2\u0659\2\u065b\2\u0661\2\u0670\2\u06d5\2\u06d7\2\u06de\2\u06e3"+
		"\2\u06ea\2\u06ef\2\u06f1\2\u06fc\2\u06fe\2\u0701\2\u0701\2\u0712\2\u0741"+
		"\2\u074f\2\u07b3\2\u07cc\2\u07ec\2\u07f6\2\u07f7\2\u07fc\2\u07fc\2\u0802"+
		"\2\u0819\2\u081c\2\u082e\2\u0842\2\u085a\2\u0862\2\u086c\2\u08a2\2\u08b6"+
		"\2\u08b8\2\u08bf\2\u08d6\2\u08e1\2\u08e5\2\u08eb\2\u08f2\2\u093d\2\u093f"+
		"\2\u094e\2\u0950\2\u0952\2\u0957\2\u0965\2\u0973\2\u0985\2\u0987\2\u098e"+
		"\2\u0991\2\u0992\2\u0995\2\u09aa\2\u09ac\2\u09b2\2\u09b4\2\u09b4\2\u09b8"+
		"\2\u09bb\2\u09bf\2\u09c6\2\u09c9\2\u09ca\2\u09cd\2\u09ce\2\u09d0\2\u09d0"+
		"\2\u09d9\2\u09d9\2\u09de\2\u09df\2\u09e1\2\u09e5\2\u09f2\2\u09f3\2\u09fe"+
		"\2\u09fe\2\u0a03\2\u0a05\2\u0a07\2\u0a0c\2\u0a11\2\u0a12\2\u0a15\2\u0a2a"+
		"\2\u0a2c\2\u0a32\2\u0a34\2\u0a35\2\u0a37\2\u0a38\2\u0a3a\2\u0a3b\2\u0a40"+
		"\2\u0a44\2\u0a49\2\u0a4a\2\u0a4d\2\u0a4e\2\u0a53\2\u0a53\2\u0a5b\2\u0a5e"+
		"\2\u0a60\2\u0a60\2\u0a72\2\u0a77\2\u0a83\2\u0a85\2\u0a87\2\u0a8f\2\u0a91"+
		"\2\u0a93\2\u0a95\2\u0aaa\2\u0aac\2\u0ab2\2\u0ab4\2\u0ab5\2\u0ab7\2\u0abb"+
		"\2\u0abf\2\u0ac7\2\u0ac9\2\u0acb\2\u0acd\2\u0ace\2\u0ad2\2\u0ad2\2\u0ae2"+
		"\2\u0ae5\2\u0afb\2\u0afe\2\u0b03\2\u0b05\2\u0b07\2\u0b0e\2\u0b11\2\u0b12"+
		"\2\u0b15\2\u0b2a\2\u0b2c\2\u0b32\2\u0b34\2\u0b35\2\u0b37\2\u0b3b\2\u0b3f"+
		"\2\u0b46\2\u0b49\2\u0b4a\2\u0b4d\2\u0b4e\2\u0b58\2\u0b59\2\u0b5e\2\u0b5f"+
		"\2\u0b61\2\u0b65\2\u0b73\2\u0b73\2\u0b84\2\u0b85\2\u0b87\2\u0b8c\2\u0b90"+
		"\2\u0b92\2\u0b94\2\u0b97\2\u0b9b\2\u0b9c\2\u0b9e\2\u0b9e\2\u0ba0\2\u0ba1"+
		"\2\u0ba5\2\u0ba6\2\u0baa\2\u0bac\2\u0bb0\2\u0bbb\2\u0bc0\2\u0bc4\2\u0bc8"+
		"\2\u0bca\2\u0bcc\2\u0bce\2\u0bd2\2\u0bd2\2\u0bd9\2\u0bd9\2\u0c02\2\u0c05"+
		"\2\u0c07\2\u0c0e\2\u0c10\2\u0c12\2\u0c14\2\u0c2a\2\u0c2c\2\u0c3b\2\u0c3f"+
		"\2\u0c46\2\u0c48\2\u0c4a\2\u0c4c\2\u0c4e\2\u0c57\2\u0c58\2\u0c5a\2\u0c5c"+
		"\2\u0c62\2\u0c65\2\u0c82\2\u0c85\2\u0c87\2\u0c8e\2\u0c90\2\u0c92\2\u0c94"+
		"\2\u0caa\2\u0cac\2\u0cb5\2\u0cb7\2\u0cbb\2\u0cbf\2\u0cc6\2\u0cc8\2\u0cca"+
		"\2\u0ccc\2\u0cce\2\u0cd7\2\u0cd8\2\u0ce0\2\u0ce0\2\u0ce2\2\u0ce5\2\u0cf3"+
		"\2\u0cf4\2\u0d02\2\u0d05\2\u0d07\2\u0d0e\2\u0d10\2\u0d12\2\u0d14\2\u0d3c"+
		"\2\u0d3f\2\u0d46\2\u0d48\2\u0d4a\2\u0d4c\2\u0d4e\2\u0d50\2\u0d50\2\u0d56"+
		"\2\u0d59\2\u0d61\2\u0d65\2\u0d7c\2\u0d81\2\u0d84\2\u0d85\2\u0d87\2\u0d98"+
		"\2\u0d9c\2\u0db3\2\u0db5\2\u0dbd\2\u0dbf\2\u0dbf\2\u0dc2\2\u0dc8\2\u0dd1"+
		"\2\u0dd6\2\u0dd8\2\u0dd8\2\u0dda\2\u0de1\2\u0df4\2\u0df5\2\u0e03\2\u0e3c"+
		"\2\u0e42\2\u0e48\2\u0e4f\2\u0e4f\2\u0e83\2\u0e84\2\u0e86\2\u0e86\2\u0e89"+
		"\2\u0e8a\2\u0e8c\2\u0e8c\2\u0e8f\2\u0e8f\2\u0e96\2\u0e99\2\u0e9b\2\u0ea1"+
		"\2\u0ea3\2\u0ea5\2\u0ea7\2\u0ea7\2\u0ea9\2\u0ea9\2\u0eac\2\u0ead\2\u0eaf"+
		"\2\u0ebb\2\u0ebd\2\u0ebf\2\u0ec2\2\u0ec6\2\u0ec8\2\u0ec8\2\u0ecf\2\u0ecf"+
		"\2\u0ede\2\u0ee1\2\u0f02\2\u0f02\2\u0f42\2\u0f49\2\u0f4b\2\u0f6e\2\u0f73"+
		"\2\u0f83\2\u0f8a\2\u0f99\2\u0f9b\2\u0fbe\2\u1002\2\u1038\2\u103a\2\u103a"+
		"\2\u103d\2\u1041\2\u1052\2\u1064\2\u1067\2\u106a\2\u1070\2\u1088\2\u1090"+
		"\2\u1090\2\u109e\2\u109f\2\u10a2\2\u10c7\2\u10c9\2\u10c9\2\u10cf\2\u10cf"+
		"\2\u10d2\2\u10fc\2\u10fe\2\u124a\2\u124c\2\u124f\2\u1252\2\u1258\2\u125a"+
		"\2\u125a\2\u125c\2\u125f\2\u1262\2\u128a\2\u128c\2\u128f\2\u1292\2\u12b2"+
		"\2\u12b4\2\u12b7\2\u12ba\2\u12c0\2\u12c2\2\u12c2\2\u12c4\2\u12c7\2\u12ca"+
		"\2\u12d8\2\u12da\2\u1312\2\u1314\2\u1317\2\u131a\2\u135c\2\u1361\2\u1361"+
		"\2\u1382\2\u1391\2\u13a2\2\u13f7\2\u13fa\2\u13ff\2\u1403\2\u166e\2\u1671"+
		"\2\u1681\2\u1683\2\u169c\2\u16a2\2\u16ec\2\u16f0\2\u16fa\2\u1702\2\u170e"+
		"\2\u1710\2\u1715\2\u1722\2\u1735\2\u1742\2\u1755\2\u1762\2\u176e\2\u1770"+
		"\2\u1772\2\u1774\2\u1775\2\u1782\2\u17b5\2\u17b8\2\u17ca\2\u17d9\2\u17d9"+
		"\2\u17de\2\u17de\2\u1822\2\u1879\2\u1882\2\u18ac\2\u18b2\2\u18f7\2\u1902"+
		"\2\u1920\2\u1922\2\u192d\2\u1932\2\u193a\2\u1952\2\u196f\2\u1972\2\u1976"+
		"\2\u1982\2\u19ad\2\u19b2\2\u19cb\2\u1a02\2\u1a1d\2\u1a22\2\u1a60\2\u1a63"+
		"\2\u1a76\2\u1aa9\2\u1aa9\2\u1b02\2\u1b35\2\u1b37\2\u1b45\2\u1b47\2\u1b4d"+
		"\2\u1b82\2\u1bab\2\u1bae\2\u1bb1\2\u1bbc\2\u1be7\2\u1be9\2\u1bf3\2\u1c02"+
		"\2\u1c37\2\u1c4f\2\u1c51\2\u1c5c\2\u1c7f\2\u1c82\2\u1c8a\2\u1ceb\2\u1cee"+
		"\2\u1cf0\2\u1cf5\2\u1cf7\2\u1cf8\2\u1d02\2\u1dc1\2\u1de9\2\u1df6\2\u1e02"+
		"\2\u1f17\2\u1f1a\2\u1f1f\2\u1f22\2\u1f47\2\u1f4a\2\u1f4f\2\u1f52\2\u1f59"+
		"\2\u1f5b\2\u1f5b\2\u1f5d\2\u1f5d\2\u1f5f\2\u1f5f\2\u1f61\2\u1f7f\2\u1f82"+
		"\2\u1fb6\2\u1fb8\2\u1fbe\2\u1fc0\2\u1fc0\2\u1fc4\2\u1fc6\2\u1fc8\2\u1fce"+
		"\2\u1fd2\2\u1fd5\2\u1fd8\2\u1fdd\2\u1fe2\2\u1fee\2\u1ff4\2\u1ff6\2\u1ff8"+
		"\2\u1ffe\2\u2073\2\u2073\2\u2081\2\u2081\2\u2092\2\u209e\2\u2104\2\u2104"+
		"\2\u2109\2\u2109\2\u210c\2\u2115\2\u2117\2\u2117\2\u211b\2\u211f\2\u2126"+
		"\2\u2126\2\u2128\2\u2128\2\u212a\2\u212a\2\u212c\2\u212f\2\u2131\2\u213b"+
		"\2\u213e\2\u2141\2\u2147\2\u214b\2\u2150\2\u2150\2\u2162\2\u218a\2\u24b8"+
		"\2\u24eb\2\u2c02\2\u2c30\2\u2c32\2\u2c60\2\u2c62\2\u2ce6\2\u2ced\2\u2cf0"+
		"\2\u2cf4\2\u2cf5\2\u2d02\2\u2d27\2\u2d29\2\u2d29\2\u2d2f\2\u2d2f\2\u2d32"+
		"\2\u2d69\2\u2d71\2\u2d71\2\u2d82\2\u2d98\2\u2da2\2\u2da8\2\u2daa\2\u2db0"+
		"\2\u2db2\2\u2db8\2\u2dba\2\u2dc0\2\u2dc2\2\u2dc8\2\u2dca\2\u2dd0\2\u2dd2"+
		"\2\u2dd8\2\u2dda\2\u2de0\2\u2de2\2\u2e01\2\u2e31\2\u2e31\2\u3007\2\u3009"+
		"\2\u3023\2\u302b\2\u3033\2\u3037\2\u303a\2\u303e\2\u3043\2\u3098\2\u309f"+
		"\2\u30a1\2\u30a3\2\u30fc\2\u30fe\2\u3101\2\u3107\2\u3130\2\u3133\2\u3190"+
		"\2\u31a2\2\u31bc\2\u31f2\2\u3201\2\u3402\2\u4db7\2\u4e02\2\u9fec\2\ua002"+
		"\2\ua48e\2\ua4d2\2\ua4ff\2\ua502\2\ua60e\2\ua612\2\ua621\2\ua62c\2\ua62d"+
		"\2\ua642\2\ua670\2\ua676\2\ua67d\2\ua681\2\ua6f1\2\ua719\2\ua721\2\ua724"+
		"\2\ua78a\2\ua78d\2\ua7b0\2\ua7b2\2\ua7b9\2\ua7f9\2\ua803\2\ua805\2\ua807"+
		"\2\ua809\2\ua80c\2\ua80e\2\ua829\2\ua842\2\ua875\2\ua882\2\ua8c5\2\ua8c7"+
		"\2\ua8c7\2\ua8f4\2\ua8f9\2\ua8fd\2\ua8fd\2\ua8ff\2\ua8ff\2\ua90c\2\ua92c"+
		"\2\ua932\2\ua954\2\ua962\2\ua97e\2\ua982\2\ua9b4\2\ua9b6\2\ua9c1\2\ua9d1"+
		"\2\ua9d1\2\ua9e2\2\ua9e6\2\ua9e8\2\ua9f1\2\ua9fc\2\uaa00\2\uaa02\2\uaa38"+
		"\2\uaa42\2\uaa4f\2\uaa62\2\uaa78\2\uaa7c\2\uaa7c\2\uaa80\2\uaac0\2\uaac2"+
		"\2\uaac2\2\uaac4\2\uaac4\2\uaadd\2\uaadf\2\uaae2\2\uaaf1\2\uaaf4\2\uaaf7"+
		"\2\uab03\2\uab08\2\uab0b\2\uab10\2\uab13\2\uab18\2\uab22\2\uab28\2\uab2a"+
		"\2\uab30\2\uab32\2\uab5c\2\uab5e\2\uab67\2\uab72\2\uabec\2\uac02\2\ud7a5"+
		"\2\ud7b2\2\ud7c8\2\ud7cd\2\ud7fd\2\uf902\2\ufa6f\2\ufa72\2\ufadb\2\ufb02"+
		"\2\ufb08\2\ufb15\2\ufb19\2\ufb1f\2\ufb2a\2\ufb2c\2\ufb38\2\ufb3a\2\ufb3e"+
		"\2\ufb40\2\ufb40\2\ufb42\2\ufb43\2\ufb45\2\ufb46\2\ufb48\2\ufbb3\2\ufbd5"+
		"\2\ufd3f\2\ufd52\2\ufd91\2\ufd94\2\ufdc9\2\ufdf2\2\ufdfd\2\ufe72\2\ufe76"+
		"\2\ufe78\2\ufefe\2\uff23\2\uff3c\2\uff43\2\uff5c\2\uff68\2\uffc0\2\uffc4"+
		"\2\uffc9\2\uffcc\2\uffd1\2\uffd4\2\uffd9\2\uffdc\2\uffde\2\2\3\r\3\17"+
		"\3(\3*\3<\3>\3?\3A\3O\3R\3_\3\u0082\3\u00fc\3\u0142\3\u0176\3\u0282\3"+
		"\u029e\3\u02a2\3\u02d2\3\u0302\3\u0321\3\u032f\3\u034c\3\u0352\3\u037c"+
		"\3\u0382\3\u039f\3\u03a2\3\u03c5\3\u03ca\3\u03d1\3\u03d3\3\u03d7\3\u0402"+
		"\3\u049f\3\u04b2\3\u04d5\3\u04da\3\u04fd\3\u0502\3\u0529\3\u0532\3\u0565"+
		"\3\u0602\3\u0738\3\u0742\3\u0757\3\u0762\3\u0769\3\u0802\3\u0807\3\u080a"+
		"\3\u080a\3\u080c\3\u0837\3\u0839\3\u083a\3\u083e\3\u083e\3\u0841\3\u0857"+
		"\3\u0862\3\u0878\3\u0882\3\u08a0\3\u08e2\3\u08f4\3\u08f6\3\u08f7\3\u0902"+
		"\3\u0917\3\u0922\3\u093b\3\u0982\3\u09b9\3\u09c0\3\u09c1\3\u0a02\3\u0a05"+
		"\3\u0a07\3\u0a08\3\u0a0e\3\u0a15\3\u0a17\3\u0a19\3\u0a1b\3\u0a35\3\u0a62"+
		"\3\u0a7e\3\u0a82\3\u0a9e\3\u0ac2\3\u0ac9\3\u0acb\3\u0ae6\3\u0b02\3\u0b37"+
		"\3\u0b42\3\u0b57\3\u0b62\3\u0b74\3\u0b82\3\u0b93\3\u0c02\3\u0c4a\3\u0c82"+
		"\3\u0cb4\3\u0cc2\3\u0cf4\3\u1002\3\u1047\3\u1084\3\u10ba\3\u10d2\3\u10ea"+
		"\3\u1102\3\u1134\3\u1152\3\u1174\3\u1178\3\u1178\3\u1182\3\u11c1\3\u11c3"+
		"\3\u11c6\3\u11dc\3\u11dc\3\u11de\3\u11de\3\u1202\3\u1213\3\u1215\3\u1236"+
		"\3\u1239\3\u1239\3\u1240\3\u1240\3\u1282\3\u1288\3\u128a\3\u128a\3\u128c"+
		"\3\u128f\3\u1291\3\u129f\3\u12a1\3\u12aa\3\u12b2\3\u12ea\3\u1302\3\u1305"+
		"\3\u1307\3\u130e\3\u1311\3\u1312\3\u1315\3\u132a\3\u132c\3\u1332\3\u1334"+
		"\3\u1335\3\u1337\3\u133b\3\u133f\3\u1346\3\u1349\3\u134a\3\u134d\3\u134e"+
		"\3\u1352\3\u1352\3\u1359\3\u1359\3\u135f\3\u1365\3\u1402\3\u1443\3\u1445"+
		"\3\u1447\3\u1449\3\u144c\3\u1482\3\u14c3\3\u14c6\3\u14c7\3\u14c9\3\u14c9"+
		"\3\u1582\3\u15b7\3\u15ba\3\u15c0\3\u15da\3\u15df\3\u1602\3\u1640\3\u1642"+
		"\3\u1642\3\u1646\3\u1646\3\u1682\3\u16b7\3\u1702\3\u171b\3\u171f\3\u172c"+
		"\3\u18a2\3\u18e1\3\u1901\3\u1901\3\u1a02\3\u1a34\3\u1a37\3\u1a40\3\u1a52"+
		"\3\u1a85\3\u1a88\3\u1a99\3\u1ac2\3\u1afa\3\u1c02\3\u1c0a\3\u1c0c\3\u1c38"+
		"\3\u1c3a\3\u1c40\3\u1c42\3\u1c42\3\u1c74\3\u1c91\3\u1c94\3\u1ca9\3\u1cab"+
		"\3\u1cb8\3\u1d02\3\u1d08\3\u1d0a\3\u1d0b\3\u1d0d\3\u1d38\3\u1d3c\3\u1d3c"+
		"\3\u1d3e\3\u1d3f\3\u1d41\3\u1d43\3\u1d45\3\u1d45\3\u1d48\3\u1d49\3\u2002"+
		"\3\u239b\3\u2402\3\u2470\3\u2482\3\u2545\3\u3002\3\u3430\3\u4402\3\u4648"+
		"\3\u6802\3\u6a3a\3\u6a42\3\u6a60\3\u6ad2\3\u6aef\3\u6b02\3\u6b38\3\u6b42"+
		"\3\u6b45\3\u6b65\3\u6b79\3\u6b7f\3\u6b91\3\u6f02\3\u6f46\3\u6f52\3\u6f80"+
		"\3\u6f95\3\u6fa1\3\u6fe2\3\u6fe3\3\u7002\3\u87ee\3\u8802\3\u8af4\3\ub002"+
		"\3\ub120\3\ub172\3\ub2fd\3\ubc02\3\ubc6c\3\ubc72\3\ubc7e\3\ubc82\3\ubc8a"+
		"\3\ubc92\3\ubc9b\3\ubca0\3\ubca0\3\ud402\3\ud456\3\ud458\3\ud49e\3\ud4a0"+
		"\3\ud4a1\3\ud4a4\3\ud4a4\3\ud4a7\3\ud4a8\3\ud4ab\3\ud4ae\3\ud4b0\3\ud4bb"+
		"\3\ud4bd\3\ud4bd\3\ud4bf\3\ud4c5\3\ud4c7\3\ud507\3\ud509\3\ud50c\3\ud50f"+
		"\3\ud516\3\ud518\3\ud51e\3\ud520\3\ud53b\3\ud53d\3\ud540\3\ud542\3\ud546"+
		"\3\ud548\3\ud548\3\ud54c\3\ud552\3\ud554\3\ud6a7\3\ud6aa\3\ud6c2\3\ud6c4"+
		"\3\ud6dc\3\ud6de\3\ud6fc\3\ud6fe\3\ud716\3\ud718\3\ud736\3\ud738\3\ud750"+
		"\3\ud752\3\ud770\3\ud772\3\ud78a\3\ud78c\3\ud7aa\3\ud7ac\3\ud7c4\3\ud7c6"+
		"\3\ud7cd\3\ue002\3\ue008\3\ue00a\3\ue01a\3\ue01d\3\ue023\3\ue025\3\ue026"+
		"\3\ue028\3\ue02c\3\ue802\3\ue8c6\3\ue902\3\ue945\3\ue949\3\ue949\3\uee02"+
		"\3\uee05\3\uee07\3\uee21\3\uee23\3\uee24\3\uee26\3\uee26\3\uee29\3\uee29"+
		"\3\uee2b\3\uee34\3\uee36\3\uee39\3\uee3b\3\uee3b\3\uee3d\3\uee3d\3\uee44"+
		"\3\uee44\3\uee49\3\uee49\3\uee4b\3\uee4b\3\uee4d\3\uee4d\3\uee4f\3\uee51"+
		"\3\uee53\3\uee54\3\uee56\3\uee56\3\uee59\3\uee59\3\uee5b\3\uee5b\3\uee5d"+
		"\3\uee5d\3\uee5f\3\uee5f\3\uee61\3\uee61\3\uee63\3\uee64\3\uee66\3\uee66"+
		"\3\uee69\3\uee6c\3\uee6e\3\uee74\3\uee76\3\uee79\3\uee7b\3\uee7e\3\uee80"+
		"\3\uee80\3\uee82\3\uee8b\3\uee8d\3\uee9d\3\ueea3\3\ueea5\3\ueea7\3\ueeab"+
		"\3\ueead\3\ueebd\3\uf132\3\uf14b\3\uf152\3\uf16b\3\uf172\3\uf18b\3\2\4"+
		"\ua6d8\4\ua702\4\ub736\4\ub742\4\ub81f\4\ub822\4\ucea3\4\uceb2\4\uebe2"+
		"\4\uf802\4\ufa1f\4o\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2"+
		"\13\3\2\2\2\3!\3\2\2\2\5?\3\2\2\2\7D\3\2\2\2\tK\3\2\2\2\13R\3\2\2\2\r"+
		"\16\7f\2\2\16\"\7g\2\2\17\"\7\u00e2\2\2\20\21\7n\2\2\21\"\7g\2\2\22\23"+
		"\7n\2\2\23\"\7c\2\2\24\25\7g\2\2\25\"\7v\2\2\26\27\7k\2\2\27\"\7n\2\2"+
		"\30\31\7n\2\2\31\32\7g\2\2\32\"\7u\2\2\33\34\7w\2\2\34\"\7p\2\2\35\36"+
		"\7g\2\2\36\"\7p\2\2\37 \7f\2\2 \"\7w\2\2!\r\3\2\2\2!\17\3\2\2\2!\20\3"+
		"\2\2\2!\22\3\2\2\2!\24\3\2\2\2!\26\3\2\2\2!\30\3\2\2\2!\33\3\2\2\2!\35"+
		"\3\2\2\2!\37\3\2\2\2\"#\3\2\2\2#$\b\2\2\2$\4\3\2\2\2%&\7v\2\2&\'\7j\2"+
		"\2\'@\7g\2\2()\7q\2\2)@\7h\2\2*+\7c\2\2+,\7p\2\2,@\7f\2\2-.\7v\2\2.@\7"+
		"q\2\2/@\7c\2\2\60\61\7j\2\2\61\62\7k\2\2\62@\7u\2\2\63\64\7k\2\2\64@\7"+
		"p\2\2\65\66\7y\2\2\66\67\7k\2\2\678\7v\2\28@\7j\2\29@\7K\2\2:;\7y\2\2"+
		";<\7j\2\2<=\7k\2\2=>\7e\2\2>@\7j\2\2?%\3\2\2\2?(\3\2\2\2?*\3\2\2\2?-\3"+
		"\2\2\2?/\3\2\2\2?\60\3\2\2\2?\63\3\2\2\2?\65\3\2\2\2?9\3\2\2\2?:\3\2\2"+
		"\2@A\3\2\2\2AB\b\3\3\2B\6\3\2\2\2CE\t\2\2\2DC\3\2\2\2EF\3\2\2\2FD\3\2"+
		"\2\2FG\3\2\2\2GH\3\2\2\2HI\b\4\4\2I\b\3\2\2\2JL\t\3\2\2KJ\3\2\2\2LM\3"+
		"\2\2\2MK\3\2\2\2MN\3\2\2\2NO\3\2\2\2OP\b\5\4\2P\n\3\2\2\2QS\t\2\2\2RQ"+
		"\3\2\2\2RS\3\2\2\2ST\3\2\2\2TU\7k\2\2UV\7p\2\2VX\3\2\2\2WY\t\2\2\2XW\3"+
		"\2\2\2XY\3\2\2\2Y\f\3\2\2\2\t\2!?FMRX\5\3\2\2\3\3\3\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}