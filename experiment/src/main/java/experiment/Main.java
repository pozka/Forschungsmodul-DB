package experiment;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.collection.mutable.WrappedArray;

public class Main {

	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().appName("HelloWorld").master("local")
				.getOrCreate();

		spark.sparkContext().setLogLevel("ERROR");

		printSparkBegin();

		SQLContext context = new org.apache.spark.sql.SQLContext(spark);

		// Read CSV

		StructType schema = new StructType(new StructField[] {
				new StructField("title", DataTypes.StringType, false, Metadata.empty()),
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("date", DataTypes.StringType, false, Metadata.empty()),
				new StructField("article", DataTypes.StringType, false, Metadata.empty()),
		});

		Dataset<Row> df = context.read()
				.format("com.databricks.spark.csv")
				.schema(schema)
				.option("header", "false")
				.option("delimiter", "\t")
				.load("wiki_9_1.csv");

		df.printSchema();

		// Ausgeben der CSV Datei
		df.show();

		// Tokenize 1

		RegexTokenizer rtGetWords = new RegexTokenizer()
				.setInputCol("article")
				.setOutputCol("words")
				.setPattern("\\W");

		spark.udf().register("countTokens", (WrappedArray<?> words) -> words.size(),
				DataTypes.IntegerType);

//		spark.udf().register("x2Multiplier",
//				new UDF1<WrappedArray<?>, WrappedArray<?>>() {
//					private static final long serialVersionUID = -5372447039252716846L;

//					@Override
//					public WrappedArray<?> call(WrappedArray<?> t1) throws Exception {
//						int i =0;
//						String[] t2 = (String[]) t1.toArray(null);
//						String[] retArr = new String[t2.length];
//
//						for (String str : t2) {
//
//							if (str.charAt(0) == '['
//									&& str.charAt(1) == '['
//									&& str.charAt(str.length() - 1) == ']'
//									&& str.charAt(str.length() - 2) == ']') {
//								retArr[i++] = str.substring(2, str.length() - 2);
//							}
//						}
//						return retArr;
//					}
//				}, DataTypes.ArrayType);

		Dataset<Row> rtdAllWords = rtGetWords.transform(df);

		System.out.println("rtdAllWords - Anzahl Wörter der Revisionen");

		rtdAllWords.select("id", "article", "words")
				.withColumn("tokens", callUDF("countTokens", col("words"))).show();

		rtdAllWords.printSchema();

		// Tokenize 2

		String pattern1 = "\\x5b\\x5b[\\w\\s]*\\x5d\\x5d";
		String pattern2 = "\\x5d\\x5d[\\w\\s&&[^\\x5d\\x5b]]*\\x5b\\x5b";

		RegexTokenizer rtGetBlueWords = new RegexTokenizer()
				.setInputCol("article")
				.setOutputCol("Blue Words")
				.setGaps(false)
				.setPattern(pattern1)
				// .setPattern("[\\w\\s]*")
				.setToLowercase(false);

		Dataset<Row> rtdBlueWords = rtGetBlueWords.transform(df);

		System.out.println("rtdBlueWords - Linkwörter");

		rtdBlueWords.select("id", "article", "Blue Words")
				// .withColumn("Blue Words 2", regexp_replace(col("Blue Words"),
				// "\\x5d\\x5d", ""))
				.withColumn("tokens", callUDF("countTokens", col("Blue Words")))
				.show(false);

		// regexTokenizedBlue.withColumn("Blue Words sub", col("Blue
		// Words").substring(col("Blue Words"), int 2,))

		rtdBlueWords.printSchema();

		// rtdBlueWords.

		// erase stopwords
		
		// Einlesen der 50 Zeilen-Datei

				StructType table = new StructType(new StructField[] {
						new StructField("title", DataTypes.StringType, false, Metadata.empty()),
						new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("date", DataTypes.StringType, false, Metadata.empty()),
						new StructField("article", DataTypes.StringType, false, Metadata.empty()),
				});

				Dataset<Row> df_02 = context.read()
						.format("com.databricks.spark.csv")
						.schema(table)
						.option("header", "false")
						.option("delimiter", "\t")
						.load("wiki_9_50.csv");

				df_02.printSchema();
				df_02.show();
				
		// Anzahl Revisionen der einzelnen Artikel
				System.out.println("Anzahl der Revisionen pro Artikel: " + "\n");
				df_02.groupBy("title").count().withColumnRenamed("title", "title").show();
		
		// Anzahl Revisionen pro Autoren-ID
				System.out.println("Anzahl der Revisionen pro Autoren-ID: " + "\n");
				df_02.groupBy("id").count().withColumnRenamed("id", "id").show();
				
		// Anzahl Revisionen pro Tag
				System.out.println("Anzahl der Revisionen pro Tag: " + "\n");
				df_02.groupBy("date").count().withColumnRenamed("date", "date").show();
				
		
		spark.stop();
	}

	private static void printSparkBegin() {
		printDog();
		System.out.println("\n" + "Spark commenced" + "\n");
	}

	private static void printDog() {
		System.out.println("\n"
				+ "         ,--._______,-.\n"
				+ "       ,','  ,    .  ,_`-.\n"
				+ "      / /  ,' , _` ``. |  )       `-..\n"
				+ "     (,';'\"\"`/ '\"`-._ ` \\/ ______    \\\\\n"
				+ "       : ,o.-`- ,o.  )\\` -'      `---.))\n"
				+ "       : , d8b ^-.   '|   `.      `    `.\n"
				+ "       |/ __:_     `. |  ,  `       `    \\\n"
				+ "       | ( ,-.`-.    ;'  ;   `       :    ;\n"
				+ "       | |  ,   `.      /     ;      :    \\\n"
				+ "       ;-'`:::._,`.__),'             :     ;\n"
				+ "      / ,  `-   `--                  ;     |\n"
				+ "     /  \\                   `       ,      |\n"
				+ "    (    `     :              :    ,\\      |\n"
				+ "     \\   `.    :     :        :  ,'  \\    :\n"
				+ "      \\    `|-- `     \\ ,'    ,-'     :-.-';\n"
				+ "      :     |`--.______;     |        :    :\n"
				+ "       :    /           |    |         |   \\\n"
				+ "       |    ;           ;    ;        /     ;\n"
				+ "     _/--' |           :`-- /         \\_:_:_|\n"
				+ "   ,',','  |           |___ \\\n"
				+ "   `^._,--'           / , , .)\n"
				+ "                      `-._,-'\n");
	}

	private static String[] truncate(String[] stringArr) {
		int i = 0;
		String[] retArr = new String[stringArr.length];

		for (String str : stringArr) {

			if (str.charAt(0) == '['
					&& str.charAt(1) == '['
					&& str.charAt(str.length() - 1) == ']'
					&& str.charAt(str.length() - 2) == ']') {
				retArr[i++] = str.substring(2, str.length() - 2);
			}
		}
		return retArr;
	}

}
