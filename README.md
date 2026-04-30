My wine dataset is a predictive data to measure the quality of wine per the content of features as follows:
| Feature                  | Description                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| **fixed acidity**        | Amount of non-volatile acids (like tartaric acid). Higher values can make wine taste sharper. |
| **volatile acidity**     | Amount of acetic acid in the wine. Too much gives a vinegar taste.                            |
| **citric acid**          | Adds freshness and flavor; usually present in small amounts.                                  |
| **residual sugar**       | Amount of sugar remaining after fermentation.                                                 |
| **chlorides**            | Salt content in the wine.                                                                     |
| **free sulfur dioxide**  | Free SO₂ that prevents microbial growth and oxidation.                                        |
| **total sulfur dioxide** | Total SO₂ (free + bound). Helps preserve wine.                                                |
| **density**              | Density of the wine (related to sugar and alcohol content).                                   |
| **pH**                   | Measures acidity/alkalinity of the wine.                                                      |
| **sulphates**            | Potassium sulphate; contributes to preservation and flavor.                                   |
| **alcohol**              | Alcohol percentage in the wine.                                                               |
| **quality**              | Score (usually 0–10) given by wine experts. This is the **target variable** in ML models.     |

After the analysis, the data was deployed into an app using streamlit
