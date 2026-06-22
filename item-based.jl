include("utils.jl")
using Statistics

function matriz_esparsa(dataset)
    R = spzeros(Float64, length(usuários), length(itens))
    for (usuário, item, avaliação) in dataset
        R[usuário, item] = avaliação
    end
    return R
end

function similaridade_cosseno(R)
    normas = [norm(R[:, j]) for j in 1:size(R, 2)]
    S = Matrix(R' * R)
    for i in axes(S, 1), j in axes(S, 2)
        S[i, j] = (normas[i] * normas[j]) == 0 ? 0.0 : S[i, j] / (normas[i] * normas[j])
    end
    return S
end

function predict(R, S, medias, usuário, item; k)
    linha_usuário = R[usuário, :]
    avaliações_usuário = setdiff(findall(!iszero, linha_usuário), [item])

    isempty(avaliações_usuário) && return medias[item]

    vizinhos = [(filme, S[item, filme]) for filme in avaliações_usuário]

    vizinhos = sort(vizinhos, by=x -> -abs(x[2]))
    vizinhos = vizinhos[1:min(k, length(vizinhos))]

    soma_média_ponderada = 0.0
    soma_pesos = 0.0

    for (filme, similaridade) in vizinhos
        soma_média_ponderada += similaridade * R[usuário, filme]
        soma_pesos += abs(similaridade)
    end

    return soma_pesos == 0 ? medias[item] : soma_média_ponderada / soma_pesos
end

function damped_means(R, μ, β)
    n_itens = size(R, 2)

    medias = zeros(n_itens)

    for item in 1:n_itens
        notas = nonzeros(R[:, item])
        n = length(notas)

        medias[item] = n == 0 ? μ : (sum(notas) + β * μ) / (n + β)
    end

    medias
end

function agrupar_usuarios(registros)

    usuarios = Dict{Int,Vector{Tuple{Int,Float64,Float64}}}()

    for (u, i, r, r̂) in registros
        push!(get!(usuarios, u, []), (i, r, r̂))
    end

    return usuarios
end


function precision_at_n(registros, N; threshold=4)

    usuarios = agrupar_usuarios(registros)

    precisions = Float64[]

    for itens in values(usuarios)

        relevantes = Set(i for (i, r, _) in itens if r >= threshold)

        isempty(relevantes) && continue

        ordenados = sort(itens, by=x -> x[3], rev=true)

        topN = ordenados[1:min(N, length(ordenados))]

        recomendados = Set(i for (i, _, _) in topN)

        hits = length(intersect(relevantes, recomendados))

        push!(precisions, hits / length(recomendados))
    end

    return mean(precisions)
end

function recall_at_n(registros, N; threshold=4)

    usuarios = agrupar_usuarios(registros)

    recalls = Float64[]

    for itens in values(usuarios)

        relevantes = Set(i for (i, r, _) in itens if r >= threshold)

        isempty(relevantes) && continue

        ordenados = sort(itens, by=x -> x[3], rev=true)

        topN = ordenados[1:min(N, length(ordenados))]

        recomendados = Set(i for (i, _, _) in topN)

        hits = length(intersect(relevantes, recomendados))

        push!(recalls, hits / length(relevantes))
    end

    return mean(recalls)
end

function f1_at_n(registros, N; threshold=4)

    precision = precision_at_n(registros, N; threshold=threshold)

    recall = recall_at_n(registros, N; threshold=threshold)

    precision + recall == 0 && return 0.0

    return 2 * precision * recall / (precision + recall)
end

function testar_modelo(R, S, medias, dataset, k)
    preditos = Float64[]
    reais = Float64[]
    registros = []

    for (usuário, item, avaliação) in dataset

        r̂ = predict(R, S, medias, usuário, item; k=k)

        push!(preditos, r̂)
        push!(reais, avaliação)
        push!(registros, (usuário, item, avaliação, r̂))
    end

    return rmse(preditos, reais), mae(preditos, reais), registros

end

function testar_parâmetros(dataset, R, S, μ, ks, βs)
    resultados = []
    melhor_rmse = Inf
    melhor_k = 0
    melhor_β = 0
    for k in ks
        for β in βs
            medias = damped_means(R, μ, β)

            rmse_val, mae_val = testar_modelo(R, S, medias, dataset, k)

            push!(resultados, (k=k, β=β, rmse=rmse_val, mae=mae_val))

            println("k=$k β=$β RMSE=$(round(rmse_val, digits=4))")

            if rmse_val < melhor_rmse
                melhor_rmse = rmse_val
                melhor_k = k
                melhor_β = β
            end
        end
    end

    return melhor_k, melhor_β, resultados
end

rmse(preditos, reais) = sqrt(mean((preditos .- reais) .^ 2))
mae(preditos, reais) = mean(abs.(preditos .- reais))
