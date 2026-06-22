include("item-based.jl")
include("graficos.jl")

u_data, usuários, itens = load_data()
treino, teste, validação = split_treino_teste_validação(u_data)

ks = [10, 20, 30, 50, 75, 100]
βs = [0, 5, 10, 25, 50, 100]

R = matriz_esparsa(treino)
S = similaridade_cosseno(R)
μ = mean(nonzeros(R))

k, β, resultados_parametros = testar_parâmetros(validação, R, S, μ, ks, βs)
medias = damped_means(R, μ, β)

plot_RMSE_k(resultados_parametros, ks, "item-based")
plot_damped_mean(resultados_parametros, βs, "item-based")

rmse_teste, mae_teste, resultados_teste = testar_modelo(R, S, medias, teste, k)
p10 = precision_at_n(resultados_teste, 10)
r10 = recall_at_n(resultados_teste, 10)
f10 = f1_at_n(resultados_teste, 10)

println("===== RESULTADO FINAL TESTE =====")
println("k = $k")
println("β = $β")
println("RMSE = $(round(rmse_teste, digits=4))")
println("MAE = $(round(mae_teste, digits=4))")
println("Precision@10 = $(round(p10, digits=4))")
println("Recall@10 = $(round(r10, digits=4))")
println("F1@10 = $(round(f10, digits=4))")

plot_precision_recall_f1_at_n(resultados_teste, "item-based")
