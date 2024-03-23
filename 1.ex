Mix.install([
  {:axon, "~> 0.5"},     # nn
  {:nx, "~> 0.5"},       # tensors
  {:explorer, "~> 0.5"}, # dataframes
  {:kino,     "~> 0.8"}  # livebook renderer
])

require Explorer.DataFrame, as: DF

iris = Explorer.Datasets.iris()

cols = ~w(sepal_width sepal_length petal_length petal_width)
normalized_iris =
  DF.mutate(
    iris,
    for col <- across(^cols) do
      {col.name, (col - mean(col)) / variance(col)}
    end)

normalized_iris =
  DF.mutate(normalized_iris, [species: Explorer.Series.cast(species, :category)])

shuffled_normalized_iris = DF.shuffle(normalized_iris)

train_df = DF.slice(shuffled_normalized_iris, 0..119)
test_df  = DF.slice(shuffled_normalized_iris, 120..149)

feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
x_train = Nx.stack(train_df[feature_columns], axis: -1)
y_train =
  (train_df["species"]
   |> Nx.stack(axis: -1)
   |> Nx.equal(Nx.iota({1, 3}, axis: -1)))
x_test = Nx.stack(test_df[feature_columns], axis: -1)
y_test =
  (test_df["species"]
  |> Nx.stack(axis: -1)
  |> Nx.equal(Nx.iota({1, 3},  axis: -1)))

model = Axon.input("iris_features", shape: {nil, 4}) |> Axon.dense(3, activation: :softmax)

Axon.Display.as_graph(model, Nx.template({1, 4}, :f32))
# iris-features {1, 4} -> dense {1, 3} -> softmax {1, 3}

data_stream = Stream.repeatedly(fn -> {x_train, y_train} end)

# batch size?
trained_model_state =
  (model
   |> Axon.Loop.trainer(:categorical_cross_entropy, :sgd)
   |> Axon.Loop.metric(:accuracy)
   |> Axon.Loop.run(data_stream, %{}, iterations: 500, epochs: 10))

data = [{x_test, y_test}]

(model
 |> Axon.Loop.evaluator()
 |> Axon.Loop.metric(:accuracy)
 |> Axon.Loop.run(data, trained_model_state))
