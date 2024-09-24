using MLJ
using MLJBase
using RDatasets
using StatsBase
using NearestNeighbors

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function predict_knn(kdtree, train, x, k)
    ids, dists = knn(kdtree, x, k)
    labels = train[:, 5][ids]
    wts = 1 ./ dists
    label_counts = Dict{eltype(labels), Float64}()
    for (label, wt) in zip(labels, wts)
        label_counts[label] = get(label_counts, label, 0.0) + wt
    end
    return argmax(label_counts)
end

iris = shuffle(dataset("datasets", "iris"))

train, test = splitdf(iris,0.8)
train_numeric = Matrix(train[:, 1:4])
test_numeric = Matrix(test[:, 1:4])

kdtree = KDTree(train_numeric)

y_pred = [predict_knn(kdtree,train, x, 4) for x in eachrow(test_numeric)]

y_true = test[:, 5]
accuracy = mean(y_true .== y_pred)
println("Accuracy: ", accuracy)