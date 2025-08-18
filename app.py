#%%
 
import streamlit as st

st.markdown('# Descubra a Felicidade')


redes_opt = ['LinkedIn', 'Twitch', 'YouTube', 'Instagram', 'Amigos',
       'Twitter / X', 'Outra rede social']
cursos_opt = ['0', '1', '2', '3', 'Mais que 3']


redes = st.selectbox('Como conheceu o Téo Me Why?', options=redes_opt)
cursos = st.selectbox('Quantos cursos acompanhou do Téo Me Why?', options=cursos_opt)




col1, col2, col3 = st.columns(3)

with col1:
    video_game = st.radio('Curte video games?', ['Sim', 'Não'])
    futebol = st.radio('Curte futebol?', ['Sim', 'Não'])

with col2:
    livros = st.radio('Curte livros?', ['Sim', 'Não'])
    tabuleiro = st.radio('Curte jogos de tabuleiro?', ['Sim', 'Não'])

with col3:
    f1 = st.radio('Curte jogos de Fórmula 1?', ['Sim', 'Não'])
    video_game = st.radio('Curte games?', ['Sim', 'Não'])


idade = st.number_input('Sua idade', 18, 100)

num_vars = ['Curte games?',
            'Curte futebol?',
            'Curte livros?',
            'Curte jogos de tabuleiro?',
            'Curte jogos de fórmula 1?',
            'Curte jogos de MMA?',
            'Idade'
]

estado_opt = ['AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MT', 'PA', 'PB',
       'PE', 'PR', 'RJ', 'RN', 'RS', 'SC', 'SP']
formacao_opt = ['Exatas', 'Biológicas', 'Humanas']

estado = st.selectbox('Estado que mora atualmente', options=estado_opt)
formacao = st.selectbox('Área de Formação', options=formacao_opt)
    # 'Tempo que atua na área de dados',
    # 'Posição da cadeira (senioridade)'





# %%
