# 🌱 Zona Agroecológica - Estimativa de Produtividade Potencial e Atingível

Aplicativo interativo em **Streamlit** para estimativa da **produtividade potencial (FAO/AEZ)** e da **produtividade atingível** considerando o balanço hídrico climatológico (BHC).

## ✨ Funcionalidades
- Download automático de dados climáticos médios (NASA POWER, 1991-2020).
- Cálculo da evapotranspiração de referência (ETo) pelos métodos:
  - Penman simplificado
  - Thornthwaite
- Simulação do balanço hídrico climatológico (BHC).
- Estimativa da produtividade potencial da cultura (FAO/AEZ).
- Correção da produtividade para déficit hídrico (produtividade atingível).
- Configuração dos coeficientes de cultivo (Kc) e sensibilidade (Ky) por fases.
- Visualização de tabelas e gráficos interativos.
- Exportação dos resultados em Excel.

## 🚀 Como executar localmente
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/zona-agro.git
   cd zona-agro
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o aplicativo:
   ```bash
   streamlit run zona.py
   ```

## ☁️ Deploy
O app pode ser publicado gratuitamente no [Streamlit Community Cloud](https://streamlit.io/cloud).

## 📄 Licença
Projeto desenvolvido para fins acadêmicos e didáticos pelo **Prof. Cláudio Ricardo da Silva (UFU)**.
