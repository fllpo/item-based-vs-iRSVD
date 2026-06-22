using Plots

function plot_RMSE_k(resultados_parametro, ks, nome_modelo)

    ks_plot = Int[]
    rmse_plot = Float64[]

    for k in ks

        rmses_k = [r.rmse for r in resultados_parametro if r.k == k]

        push!(ks_plot, k)
        push!(rmse_plot, minimum(rmses_k))
    end

    p = plot(ks_plot,
        rmse_plot,
        xlabel="Número de vizinhos (k)",
        ylabel="RMSE",
        label="Validação", linewidth=2,
        title="RMSE em função de k")

    savefig(p, "graficos/$nome_modelo/RMSE_k.pdf")
end

function plot_damped_mean(resultados_parametro, βs, nome_modelo)
    betas_plot = Int[]
    rmse_plot = Float64[]

    for β in βs

        rmses_β = [r.rmse for r in resultados_parametro if r.β == β]

        push!(betas_plot, β)
        push!(rmse_plot, minimum(rmses_β))
    end

    p = plot(betas_plot,
        rmse_plot,
        xlabel="β",
        ylabel="RMSE",
        label="Validação",
        title="Impacto do β em Damped Mean ($nome_modelo)")

    savefig(p, "graficos/$nome_modelo/damped_means.pdf")
end

function plot_distribuição_reais_preditas(registros, nome_modelo)
    reais = [r for (_, _, r, _) in registros]
    preditos = [r̂ for (_, _, _, r̂) in registros]

    histogram(
        reais,
        alpha=0.5,
        bins=20,
        label="Reais",
        title="Distribuição das notas reais e preditas"
    )

    p = histogram!(
        preditos,
        alpha=0.5,
        bins=20,
        label="Preditas"
    )

    savefig(p, "graficos/$nome_modelo/distribuição_reais_preditos.pdf")
end

function plot_distribuição_erro(registros, nome_modelo)
    erros = [r̂ - r for (_, _, r, r̂) in registros]

    p = histogram(
        erros,
        bins=40,
        xlabel="Erro",
        ylabel="Frequência",
        label="",
        title="Distribuição dos erros ($nome_modelo)"
    )
    savefig(p, "graficos/$nome_modelo/distribuição_erro.pdf")
end

function plot_precision_recall_f1_at_n(registros, nome_modelo)

    Ns = [1, 3, 5, 10, 20]

    precisions = Float64[]
    recalls = Float64[]
    f1s = Float64[]

    for N in Ns
        push!(precisions, precision_at_n(registros, N))
        push!(recalls, recall_at_n(registros, N))
        push!(f1s, f1_at_n(registros, N))
    end

    plot(Ns,
        precisions,
        label="Precision@N",
        xlabel="N",
        title="Precision@N, Recall@N e F1@N  ($nome_modelo)")

    plot!(Ns,
        recalls,
        label="Recall@N")
    p = plot!(Ns, f1s, label="F1@N", xlabel="N")

    savefig(p, "graficos/$nome_modelo/precision_recall_f1_at_n.pdf")
end


function plot_iRSVD(erros, nome_modelo; val=nothing)
    p=plot(erros, xlabel="Época", ylabel="Erro", label="Treino", linewidth=2, title="Curva de Erro ($nome_modelo)")

    if val!==nothing
        plot!(p, val, label="Validação", linewidth=2)
        savefig(p, "graficos/$nome_modelo/loss_val_iRSVD.pdf")
    else
        savefig(p, "graficos/$nome_modelo/loss_iRSVD.pdf")
    end

end
