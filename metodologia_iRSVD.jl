include("iRSVD.jl")
include("graficos.jl")

u_data, usuários, itens = load_data()
treino, teste, validação = split_treino_teste_validação(u_data)

epochs = 200
λs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
γs = [0.1, 0.01, 0.001]
ks = [3, 5, 10, 20, 50, 100]

# Tuning
(k, λ, γ) = testar_hiperparâmetros(treino, validação, λs, γs, ks, epochs)
modelo_tuning = RSVD(k, mean((t[3] for t in treino)))
erros_treino, erros_validação = train!(modelo_tuning, treino, λ, γ, epochs; validação=validação)
plot_iRSVD(erros_treino, "iRSVD"; val=erros_validação)

# Treino
dado_final = vcat(treino, validação)
modelo_final = RSVD(k, mean((d[3] for d in dado_final)))
erros_final=train!(modelo_final, dado_final, λ, γ, epochs)
plot_iRSVD(erros_final, "iRSVD")

# Teste
registros = testar_modelo(modelo_final, teste)
plot_precision_recall_f1_at_n(registros, "iRSVD")

println("===== RESULTADO FINAL TESTE =====")
println("k = $k")
println("λ = $λ")
println("γ = $γ")
println("RMSE = $(round(rmse(modelo_final, teste), digits=4))")
println("MAE = $(round(mae(modelo_final, teste), digits=4))")
println("Precision@10 = $(round(precision_at_n(registros,10), digits=4))")
println("Recall@10 = $(round(recall_at_n(registros,10), digits=4))")
println("F1@10 = $(round(f1_at_n(registros,10), digits=4))")