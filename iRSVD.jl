include("utils.jl")

struct RSVD
    p::Matrix{Float64}
    q::Matrix{Float64}
    bu::Vector{Float64}
    bi::Vector{Float64}
    μ::Float64
    function RSVD(k, μ)
        p = 0.1 .* randn(length(usuários), k)
        q = 0.1 .* randn(length(itens), k)
        bu = zeros(length(usuários))
        bi = zeros(length(itens))

        new(p, q, bu, bi, μ)
    end
end

mae(modelo, dataset) = mean(abs(r - predict(modelo, u, i)) for (u, i, r) in dataset)
rmse(modelo, dataset) = sqrt(mean((r - predict(modelo, u, i))^2 for (u, i, r) in dataset))

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

function predict(modelo::RSVD, usuário, item)
    modelo.μ + modelo.bu[usuário] + modelo.bi[item] +
    dot(view(modelo.p, usuário, :), view(modelo.q, item, :))
end

function objetivo(modelo, dataset, λ)
    loss = 0.0
    for (u, i, r) in dataset
        r̂ = predict(modelo, u, i)
        loss += (r - r̂)^2
    end
    regularização = λ * (norm(modelo.bu)^2 + norm(modelo.bi)^2 +
                         norm(modelo.p)^2 + norm(modelo.q)^2)

    return loss + length(dataset) * regularização
end

function update!(modelo, dataset, λ, γ)

    idx = shuffle(1:length(dataset))

    @views for k in idx

        usuário, item, r = dataset[k]

        e = r - predict(modelo, usuário, item)

        modelo.bu[usuário] += γ * (e - λ * modelo.bu[usuário])
        modelo.bi[item] += γ * (e - λ * modelo.bi[item])

        pu = copy(modelo.p[usuário, :])
        qi = copy(modelo.q[item, :])

        modelo.p[usuário, :] += γ .* (e .* qi .- λ .* pu)
        modelo.q[item, :] += γ .* (e .* pu .- λ .* qi)

    end
end

function train!(modelo, treino, λ, γ, epochs; validação=nothing)

    erros_treino = Float64[]
    erros_validação = Float64[]

    melhor_loss_val = Inf
    melhor_p = copy(modelo.p)
    melhor_q = copy(modelo.q)
    melhor_bu = copy(modelo.bu)
    melhor_bi = copy(modelo.bi)
    épocas_sem_melhora = 0
    patience=10

    for epoch = 1:epochs
        update!(modelo, treino, λ, γ)

        loss_treino = objetivo(modelo, treino, λ) / length(treino)
        push!(erros_treino, loss_treino)

        print("Época: $epoch | loss_treino: $(round(loss_treino, digits=4))")

        if validação !== nothing
            loss_val = objetivo(modelo, validação, λ) / length(validação)
            push!(erros_validação, loss_val)

            println(" | loss_validação: $(round(loss_val,digits=4))")

            if loss_val < melhor_loss_val
                melhor_loss_val = loss_val
                melhor_p .= modelo.p
                melhor_q .= modelo.q
                melhor_bu .= modelo.bu
                melhor_bi .= modelo.bi
                épocas_sem_melhora = 0
            else
                épocas_sem_melhora += 1

                if épocas_sem_melhora>=patience
                    break
                end
            end
        else
            println()
        end
    end

    if validação!==nothing
        modelo.p .= melhor_p
        modelo.q .= melhor_q
        modelo.bu .= melhor_bu
        modelo.bi .= melhor_bi
        return (erros_treino, erros_validação)
    else
        return erros_treino
    end
end

function testar_hiperparâmetros(treino, validação, lambdas, gammas, ks, epochs)

    melhor_loss = Inf
    melhor_k = 0
    melhor_λ = 0.0
    melhor_γ = 0.0

    total = length(ks) * length(lambdas) * length(gammas)
    atual = 1

    for k in ks
        for λ in lambdas
            for γ in gammas

                println()
                println("[$atual/$total]")
                println("k = $k | λ = $λ | γ = $γ")

                modelo = RSVD(k, mean(t[3] for t in treino))
                train!(modelo, treino, λ, γ, epochs; validação=validação)

                loss_val = objetivo(modelo, validação, λ) / length(validação)
                rmse_val = rmse(modelo, validação)
                mae_val = mae(modelo, validação)

                println("Validação | Loss=$(round(loss_val, digits=4)) | RMSE=$(round(rmse_val, digits=4)) | MAE=$(round(mae_val,digits=4))")

                if loss_val < melhor_loss
                    melhor_loss = loss_val
                    melhor_k = k
                    melhor_λ = λ
                    melhor_γ = γ
                end
                atual += 1
            end
        end
    end

    return (k=melhor_k, λ=melhor_λ, γ=melhor_γ)
end

function testar_modelo(modelo, dataset)

    registros = []

    for (u, i, r) in dataset
        r̂ = predict(modelo, u, i)
        push!(registros, (u, i, r, r̂))
    end

    return registros
end