import Pkg
Pkg.add("ZipFile")
Pkg.add("DataStructures")
using DataFrames, CSV, Downloads, ZipFile, SparseArrays, LinearAlgebra, Statistics, Random

function load_data()
    z = ZipFile.Reader(Downloads.download(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip"))

    u_data = CSV.read(first(filter(f -> f.name == "ml-100k/u.data", z.files)),
        DataFrame, delim='\t', header=[:user, :id, :rating, :timestamp])

    usuários = sort(unique(u_data.user))
    itens = sort(unique(u_data.id))

    return u_data, usuários, itens
end

function split_treino_teste_validação(data, test_size=0.3, val_size=0.1)
    Random.seed!(42)
    avaliações = shuffle(collect(zip(data.user, data.id, data.rating)))
    n_teste = Int(length(avaliações) * test_size)
    n_val = Int(length(avaliações) * val_size)
    teste = avaliações[1:n_teste]
    validação = avaliações[n_teste+1:n_teste+n_val]
    treino = avaliações[n_teste+n_val+1:end]
    return treino, teste, validação
end
