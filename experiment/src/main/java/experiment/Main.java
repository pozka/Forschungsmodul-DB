package experiment;

import static org.apache.spark.sql.functions.*;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
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
				.load("wiki_9_50.csv");

		df.printSchema();

		df.show();

		// Tokenize 1

		RegexTokenizer rtGetWords = new RegexTokenizer()
				.setInputCol("article")
				.setOutputCol("words")
				.setPattern("\\W");

		spark.udf().register("countTokens", (WrappedArray<?> words) -> words.size(),
				DataTypes.IntegerType);

		Dataset<Row> rtdAllWords = rtGetWords.transform(df);

		System.out.println("rtdAllWords - Quantity of revisions");

		rtdAllWords.select("id", "article", "words")
				.withColumn("tokens", callUDF("countTokens", col("words"))).show();

		rtdAllWords.printSchema();

		// Tokenize 2

		String pattern1 = "\\x5b\\x5b[\\w\\s]*\\x5d\\x5d";

		RegexTokenizer rtGetBlueWords = new RegexTokenizer()
				.setInputCol("article")
				.setOutputCol("Blue Words")
				.setGaps(false)
				.setPattern(pattern1)
				.setToLowercase(false);

		Dataset<Row> rtdBlueWords = rtGetBlueWords.transform(df);

		System.out.println("rtdBlueWords - Bluewords");

		rtdBlueWords.select("id", "article", "Blue Words")
				.withColumn("tokens", callUDF("countTokens", col("Blue Words")))
				.show(false);

		rtdBlueWords.printSchema();

		// Blue Words explode and remove [[...]]

		Dataset<Row> rtdBlueWordsExploded = rtdBlueWords.select(
				rtdBlueWords.col("title"),
				rtdBlueWords.col("id"),
				rtdBlueWords.col("date"),
				org.apache.spark.sql.functions
						.explode(rtdBlueWords.col("Blue Words"))
						.as("Blue Words Exploded"));

		Column a = rtdBlueWordsExploded.col("Blue Words Exploded");
		a = org.apache.spark.sql.functions
				.ltrim(a, "[[");
		a = org.apache.spark.sql.functions
				.rtrim(a, "]]");
		
		// Kommentar
		Dataset<Row> rtdBlueWordsExploded2 = rtdBlueWordsExploded.select(
				rtdBlueWordsExploded.col("title"),
				rtdBlueWordsExploded.col("id"),
				rtdBlueWordsExploded.col("date"),
				a.as("Blue Words"));
		System.out.println("List Linkwords");
		rtdBlueWordsExploded2.printSchema();
		rtdBlueWordsExploded2.show();

		// Kommentar
		Dataset<Row> crossJoinedDF = rtdBlueWordsExploded2
				.withColumnRenamed("title", "title_1")
				.withColumnRenamed("id", "id_1")
				.withColumnRenamed("date", "date_1")
				.withColumnRenamed("Blue Words", "BlueWords_1")
				.crossJoin(rtdBlueWordsExploded2.withColumnRenamed("title", "title_2")
						.withColumnRenamed("id", "id_2")
						.withColumnRenamed("date", "date_2")
						.withColumnRenamed("Blue Words", "BlueWords_2"));
		crossJoinedDF.show();
		
		Dataset<Row> crossJoinedDensity = rtdBlueWordsExploded2;
		
		crossJoinedDF = crossJoinedDF.filter(col("id_1").notEqual(col("id_2")));
		
		// Kommentar
		Dataset<Row> BWtogether = crossJoinedDF
				.groupBy("id_1","id_2")
				.count()
				.withColumnRenamed("count", "BW per article");
		BWtogether.show();
		
		
//		//..
//		Dataset<Row> BWperrevision = rtdBlueWordsExploded
//				.groupBy("count")
//				.count()
//				.withColumnRenamed("count", "BW per revision");
//		BWperrevision.sort("date");
//		System.out.println("Anzahl der Linkwörter pro Revision nach Änderungsdatum geordnet");
//		BWperrevision.show();
		
//		Dataset<Row> commonBW = crossJoinedDF
//				.groupBy("Blue Words_1","Blue Words_2");

		// Blue words regroup by id

		rtdBlueWordsExploded2.groupBy("title", "id", "date")
				.agg(collect_set("Blue Words")).show(false);
		
		
		//LinkDensity as article grows
		
		Column quantity = rtdBlueWordsExploded.col("Blue Words Exploded");
		quantity = org.apache.spark.sql.functions.ltrim(quantity, "[[");
		quantity = org.apache.spark.sql.functions.rtrim(quantity, "]]");
		//int quantity_num =counter(quantity);
		
		
		//Dataset<Row> crossJoinedLD = rtdBlueWordsExploded2
		//		.crossJoin(BWtogether.withColumnRenamed("BW per article", "BW counter")).sort("date");
		
		Dataset<Row> linkdensityDS = rtdBlueWordsExploded2.select(
				rtdBlueWordsExploded2.col("title"),
				rtdBlueWordsExploded2.col("id"),
				rtdBlueWordsExploded2.col("date"));
		
		linkdensityDS.printSchema();
		linkdensityDS.show();
		
		System.out.println("Bluewords quantity //probably trash");
		Dataset<Row> crossJoinedLD = linkdensityDS
				.crossJoin(BWtogether.withColumnRenamed("BW per article", "BW quantity")).sort("date");
		crossJoinedLD.show();

		
		Dataset<Row> BWCounting = crossJoinedDensity
				.groupBy("title","date")
				.count();
		System.out.println("Bluewords quantity sorted by date of revision");
		BWCounting.sort("date").show(50);
		
		
		// Einlesen der 50 Zeilen-Datei

		// StructType table = new StructType(new StructField[] {
		// new StructField("title", DataTypes.StringType, false,
		// Metadata.empty()),
		// new StructField("id", DataTypes.IntegerType, false,
		// Metadata.empty()),
		// new StructField("date", DataTypes.StringType, false,
		// Metadata.empty()),
		// new StructField("article", DataTypes.StringType, false,
		// Metadata.empty()),
		// });
		//
		// Dataset<Row> df_02 = context.read()
		// .format("com.databricks.spark.csv")
		// .schema(table)
		// .option("header", "false")
		// .option("delimiter", "\t")
		// .load("wiki_9_50.csv");
		//
		// df_02.printSchema();
		// df_02.show();
		//
		// // Anzahl Revisionen der einzelnen Artikel
		// System.out.println("Anzahl der Revisionen pro Artikel: " + "\n");
		// df_02.groupBy("title").count().withColumnRenamed("title",
		// "title").show();
		//
		// // Anzahl Revisionen pro Autoren-ID
		// System.out.println("Anzahl der Revisionen pro Autoren-ID: " + "\n");
		// df_02.groupBy("id").count().withColumnRenamed("id", "id").show();
		//
		// // Anzahl Revisionen pro Tag
		// System.out.println("Anzahl der Revisionen pro Tag: " + "\n");
		// df_02.groupBy("date").count().withColumnRenamed("date",
		// "date").show();

		spark.stop();
	}
	
	private static int counter(String quantity) {
		
		return quantity.trim().split("\\s+").length;
	}

	private static void printSparkBegin() {
		// printDog();
		System.out.println("\n" + "Spark commenced" + "\n");
	}

//	private static void printDog() {
//		System.out.println("\n"
//				+ "         ,--._______,-.\n"
//				+ "       ,','  ,    .  ,_`-.\n"
//				+ "      / /  ,' , _` ``. |  )       `-..\n"
//				+ "     (,';'\"\"`/ '\"`-._ ` \\/ ______    \\\\\n"
//				+ "       : ,o.-`- ,o.  )\\` -'      `---.))\n"
//				+ "       : , d8b ^-.   '|   `.      `    `.\n"
//				+ "       |/ __:_     `. |  ,  `       `    \\\n"
//				+ "       | ( ,-.`-.    ;'  ;   `       :    ;\n"
//				+ "       | |  ,   `.      /     ;      :    \\\n"
//				+ "       ;-'`:::._,`.__),'             :     ;\n"
//				+ "      / ,  `-   `--                  ;     |\n"
//				+ "     /  \\                   `       ,      |\n"
//				+ "    (    `     :              :    ,\\      |\n"
//				+ "     \\   `.    :     :        :  ,'  \\    :\n"
//				+ "      \\    `|-- `     \\ ,'    ,-'     :-.-';\n"
//				+ "      :     |`--.______;     |        :    :\n"
//				+ "       :    /           |    |         |   \\\n"
//				+ "       |    ;           ;    ;        /     ;\n"
//				+ "     _/--' |           :`-- /         \\_:_:_|\n"
//				+ "   ,',','  |           |___ \\\n"
//				+ "   `^._,--'           / , , .)\n"
//				+ "                      `-._,-'\n");
//	}

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
