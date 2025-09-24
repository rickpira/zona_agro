# zona.py
# Programa para estimativa da produtividade potencial (FAO/AEZ)
# + balanço hídrico climatológico (BHC) em abas
# Desenvolvido pelo Prof. Cláudio Ricardo da Silva - UFU

import streamlit as st
import pandas as pd
import math
import ssl
import matplotlib.pyplot as plt
import socket
from datetime import date
import io, requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



st.set_page_config(
    layout="wide",
    page_title="Zona Agroecológica",
    page_icon="🌱",
)







# ------------------------------------------------------------
# BLOCO 1: Funções auxiliares (download + pré-cálculos climáticos)
# ------------------------------------------------------------
def baixar_dados(lat: float, lon: float, eto_method: str = "Penman simplificado") -> pd.DataFrame:
    # --- sessão com retry/backoff robusto ---
    session = requests.Session()
    session.headers.update({"User-Agent": "zona-agro/1.0 (contato: claudio.ricardo@ufu.br)"})

    retry = Retry(
        total=3,                 # 3 tentativas
        backoff_factor=1.5,      # 1.5s, 3s, 4.5s...
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    url = (
        f"https://power.larc.nasa.gov/api/temporal/climatology/point?"
        f"parameters=T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR&"
        f"latitude={lat}&longitude={lon}&community=AG&format=CSV&start=1991&end=2020"
    )

    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Erro de rede/DNS/timeouts/etc → propaga com mensagem clara
        raise RuntimeError(
            "Falha ao acessar a API NASA POWER (verifique conexão/DNS/VPN). "
            f"Detalhes: {type(e).__name__}: {e}"
        )

    linhas = r.text.splitlines()
    inicio = None
    for i, linha in enumerate(linhas):
        if linha.strip().startswith("PARAMETER"):
            inicio = i
            break
    df = pd.read_csv(io.StringIO("\n".join(linhas[inicio:])))

    df_mensal = df.set_index("PARAMETER").T.reset_index()
    df_mensal = df_mensal.rename(columns={
        "index": "MES",
        "ALLSKY_SFC_SW_DWN": "Qg",   # MJ m-2 d-1 (você confirmou)
        "T2M_MAX": "Tmax",
        "T2M_MIN": "Tmin",
        "RH2M": "UR",
        "WS2M": "u2",
        "PRECTOTCORR": "P"           # mm/dia
    })
    df_mensal = df_mensal[df_mensal["MES"] != "ANN"]

    # Conversões numéricas
    for col in ["Qg","Tmax","Tmin","UR","u2","P"]:
        df_mensal[col] = pd.to_numeric(df_mensal[col], errors="coerce")

    # Médias e auxiliares
    df_mensal["Tmed"] = (df_mensal["Tmax"] + df_mensal["Tmin"]) / 2
    dias_mes = [31,28,31,30,31,30,31,31,30,31,30,31]
    df_mensal["DiasMes"] = dias_mes
    df_mensal["P_total"] = df_mensal["P"] * df_mensal["DiasMes"]  # mm/mês

    # --- Astronomia: Q0 (Ra), n/N e N_h (horas de luz) ---
    def declinacao(NDA):
        ang = (360.0/365.0) * (NDA - 80.0)
        return math.radians(23.45) * math.sin(math.radians(ang))

    def angulo_por_do_sol(lat_rad, delta_rad):
        x = -math.tan(lat_rad) * math.tan(delta_rad)
        x = max(-1.0, min(1.0, x))
        return math.acos(x)  # rad

    def fator_distancia(NDA):
        return 1.0 + 0.033 * math.cos((2.0 * math.pi / 365.0) * NDA)

    def q0_diario(lat_deg, NDA):
        delta = declinacao(NDA)
        ws = angulo_por_do_sol(math.radians(lat_deg), delta)
        dr = fator_distancia(NDA)
        termo = (ws*math.sin(math.radians(lat_deg))*math.sin(delta)
                 + math.cos(math.radians(lat_deg))*math.cos(delta)*math.sin(ws))
        return 37.6 * dr * termo  # MJ/m²/dia

    dias_medios = [15,45,75,105,135,165,195,225,255,285,315,345]
    df_mensal["NDA"] = dias_medios
    df_mensal["Q0"] = [q0_diario(lat, nda) for nda in df_mensal["NDA"]]

    def n_sobre_N(Qg, Q0, a=0.25, b=0.50):
        if Q0 <= 0: return 0.0
        return max(0.0, min(1.0, (Qg/Q0 - a)/b))

    df_mensal["n/N"] = [n_sobre_N(Qg, Q0) for Qg, Q0 in zip(df_mensal["Qg"], df_mensal["Q0"])]

    ws_list = [angulo_por_do_sol(math.radians(lat), declinacao(nda)) for nda in df_mensal["NDA"]]
    df_mensal["N_h"] = [24.0 / math.pi * ws for ws in ws_list]  # horas de luz

    # --- ETo: Thornthwaite OU Penman simplificado ---
    if eto_method == "Thornthwaite":
        Tpos = df_mensal["Tmed"].clip(lower=0.0)
        I = float(((Tpos / 5.0) ** 1.514).sum())
        a = (6.75e-7)*(I**3) - (7.71e-5)*(I**2) + (1.792e-2)*I + 0.49239
        eto_mes = []
        for _, row in df_mensal.iterrows():
            T = float(row["Tmed"]); N = float(row["N_h"]); d = int(row["DiasMes"])
            if T <= 0 or I <= 0:
                eto_mensal = 0.0
            else:
                # ETo_mensal = 16 * (10*T/I)^a * (N/12) * (d/30)
                eto_mensal = 16.0 * ((10.0*T/I)**a) * (N/12.0) * (d/30.0)
            eto_mes.append(eto_mensal)
        df_mensal["ETo"] = eto_mes
    else:
        # Penman simplificado (mantido como no seu código original)
        def calc_eto(row):
            Tmean=row["Tmed"]; Tmax=row["Tmax"]; Tmin=row["Tmin"]
            RH=row["UR"]; u2=row["u2"]; Qg=row["Qg"]; Ndays=row["DiasMes"]
            es_Tmax=0.6108*math.exp((17.27*Tmax)/(Tmax+237.3))
            es_Tmin=0.6108*math.exp((17.27*Tmin)/(Tmin+237.3))
            es=(es_Tmax+es_Tmin)/2; ea=(RH/100)*es
            delta=4098*(0.6108*math.exp((17.27*Tmean)/(Tmean+237.3)))/((Tmean+237.3)**2)
            gamma=0.063
            Rn=0.55*Qg  # Qg já em MJ m-2 d-1
            eto_dia=((0.408*delta*Rn)+gamma*(900/(Tmean+273))*u2*(es-ea))/(delta+gamma*(1+0.34*u2))
            return eto_dia*Ndays
        df_mensal["ETo"] = df_mensal.apply(calc_eto, axis=1)

    # MESNUM
    mapa_meses = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    df_mensal["MESNUM"] = df_mensal["MES"].map(mapa_meses)
    
    return df_mensal



# ------------------------------------------------------------
# BLOCO 2: Balanço Hídrico (lógica exponencial corrigida)
# ------------------------------------------------------------
def bloco_umido_inicio_fim(D, tol=1e-6):
    """
    Identifica o bloco úmido (sequência de D >= 0) para definir o ponto de rotação do ano.
    """
    n = len(D)
    pos = D >= -tol
    if not np.any(pos): 
        return None, None, 'todos_neg'
    if np.all(pos): 
        return 0, n-1, 'todos_pos'

    i_ini = None
    for i in range(n):
        prev = (i-1) % n
        if (D[i] >= -tol) and (D[prev] < -tol):
            i_ini = i
            break
    if i_ini is None:
        i_ini = int(np.where(pos)[0][0])

    D2 = np.r_[D, D]
    j = i_ini
    while j < i_ini + n and D2[j] >= -tol:
        j += 1
    i_fim = (j-1) % n
    return i_ini, i_fim, 'bloco_pos'

def calcular_ARM_exponencial(P, ETo, CAD, tol=1e-5):
    """
    Thornthwaite-Mather com:
      - rotação no fim do bloco úmido,
      - ARM inicial consistente,
      - atualização exponencial na fase seca,
      - SEM arredondar durante o cálculo.
    """
    n = 12
    D = np.array(P, dtype=float) - np.array(ETo, dtype=float)

    # 1) Ponto de rotação
    i_ini, i_fim, caso = bloco_umido_inicio_fim(D, tol)
    inicio_ARM = ((int(np.argmax(D)) + 1) % n) if caso == 'todos_neg' else ((i_fim + 1) % n)

    # 2) Rotaciona
    idx_rot = [(inicio_ARM + k) % n for k in range(n)]
    D_rot = D[idx_rot]

    # 3) ARM inicial
    pos = D_rot[D_rot > 0]
    neg = D_rot[D_rot < 0]
    soma_pos = float(pos.sum()) if pos.size else 0.0
    soma_neg = float(neg.sum()) if neg.size else 0.0

    if soma_pos >= CAD - tol:
        ARM_ini = CAD
    else:
        ARM_ini = min(CAD, soma_pos) if abs(soma_neg) <= tol else (
            soma_pos / (1.0 - math.exp(soma_neg / CAD))
        )

    # 4) Atualização mês a mês (SEM arredondar)
    ARM = float(ARM_ini)
    ARM_seq_rot = []
    for d in D_rot:
        if d >= -tol:  # úmido
            ARM = min(CAD, ARM + d)
        else:          # seco (exponencial)
            delta_eq = CAD * math.log(max(ARM, 1e-12) / CAD)
            delta_eq += d
            ARM = CAD * math.exp(delta_eq / CAD)
            ARM = max(0.0, min(ARM, CAD))
        ARM_seq_rot.append(ARM)

    # 5) Desfaz rotação
    ARM_final = np.empty(n, dtype=float)
    for k, i_orig in enumerate(idx_rot):
        ARM_final[i_orig] = ARM_seq_rot[k]

    # 6) Deriva ALT, ETR, DEF, EXC (sem arredondar)
    ALT = np.zeros(n, dtype=float)
    ETR = np.zeros(n, dtype=float)
    DEF = np.zeros(n, dtype=float)
    EXC = np.zeros(n, dtype=float)

    for i in range(n):
        ip = (i - 1) % n
        ALT[i] = ARM_final[i] - ARM_final[ip]
        if D[i] >= -tol:
            recarga = max(0.0, ALT[i])
            EXC[i] = max(0.0, D[i] - recarga)
            ETR[i] = ETo[i]
            DEF[i] = 0.0
        else:
            retirada = -ALT[i]
            ETR[i] = P[i] + retirada
            DEF[i] = max(0.0, ETo[i] - ETR[i])
            EXC[i] = 0.0

    return ARM_final, ALT, ETR, DEF, EXC, inicio_ARM

def balanco_hidrico(df_mensal: pd.DataFrame, CAD: float) -> pd.DataFrame:
    # 1) P_ETo (antes de ARM)
    p_eto = df_mensal["P_total"] - df_mensal["ETo"]
    if "P_ETo" in df_mensal.columns:
        df_mensal["P_ETo"] = p_eto
        cols = list(df_mensal.columns)
        cols = [c for c in cols if c != "P_ETo"]
        epos = cols.index("ETo") + 1
        cols = cols[:epos] + ["P_ETo"] + cols[epos:]
        df_mensal = df_mensal[cols]
    else:
        epos = df_mensal.columns.get_loc("ETo") + 1
        df_mensal.insert(epos, "P_ETo", p_eto)

    # 2) BHC
    P = df_mensal["P_total"].values
    ETo = df_mensal["ETo"].values
    ARM, ALT, ETR, DEF, EXC, _ = calcular_ARM_exponencial(P, ETo, CAD)

    df_mensal["ARM"] = ARM
    df_mensal["ALT"] = ALT
    df_mensal["ETR"] = ETR
    df_mensal["DEF"] = DEF
    df_mensal["EXC"] = EXC

    return df_mensal

# ------------------------------------------------------------
# BLOCO 3: Funções FAO/AEZ (produtividade potencial)
# ------------------------------------------------------------
def CTn(T, grupo):
    if grupo==1: val=0.7+0.035*T-0.001*(T**2)
    elif grupo==2: val=0.583+0.014*T+0.0013*(T**2)-0.000037*(T**3)
    elif grupo==3: val=(-1.064+0.173*T-0.0029*(T**2)) if T>=16.5 else (-4.16+0.4325*T-0.00725*(T**2))
    return max(0,val)

def CTc(T, grupo):
    if grupo==1: val=0.25+0.0875*T-0.0025*(T**2)
    elif grupo==2: val=-0.0425+0.035*T+0.00325*(T**2)-0.0000925*(T**3)
    elif grupo==3: val=(-4.16+0.4325*T-0.00725*(T**2)) if T>=16.5 else (-9.32+0.865*T-0.0145*(T**2))
    return max(0,val)

def ppb_fao_diaria(Q0,Qg,Tmed,nN,grupo):
    ctn=CTn(Tmed,grupo); ctc=CTc(Tmed,grupo)
    PPbn=(31.7+5.2307*Q0)*ctn*(1-nN)
    PPbc=(107.2+8.5985*Q0)*ctc*nN
    return {"CTn":ctn,"CTc":ctc,"PPbn":PPbn,"PPbc":PPbc,"PPt":PPbn+PPbc}

def medias_ciclo_termico(df, mes_inicio, dia_inicio, Tb, GD_alvo):
    dias_mes_fix={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    m=mes_inicio; primeiro=True
    soma_T=soma_Qg=soma_Q0=soma_nN=0; dias_total=0; GD_total=0
    while GD_total<GD_alvo:
        ndias=dias_mes_fix[m]
        dias_usados=ndias-(dia_inicio-1) if primeiro else ndias
        row=df[df["MESNUM"]==m].iloc[0]
        Tmed=row["Tmed"]
        GD_med=max(0,Tmed-Tb)
        GD_mes=GD_med*dias_usados
        if GD_med>0 and GD_total+GD_mes>=GD_alvo:
            dias=(GD_alvo-GD_total)/GD_med
            GD_total=GD_alvo; dias_total+=dias
            soma_T+=Tmed*dias; soma_Qg+=row["Qg"]*dias; soma_Q0+=row["Q0"]*dias; soma_nN+=row["n/N"]*dias
            break
        else:
            GD_total+=GD_mes; dias_total+=dias_usados
            soma_T+=Tmed*dias_usados; soma_Qg+=row["Qg"]*dias_usados; soma_Q0+=row["Q0"]*dias_usados; soma_nN+=row["n/N"]*dias_usados
        m=1 if m==12 else m+1
        primeiro=False
    return {
        "Tmed":soma_T/dias_total,
        "Qg":soma_Qg/dias_total,
        "Q0":soma_Q0/dias_total,
        "nN":soma_nN/dias_total,
        "dias_total":dias_total,
        "GD_total":GD_total
    }

def calc_CIAF(IAF):
    return 0.5 if IAF>=5 else 0.0093+0.185*IAF-0.0175*(IAF**2)

def aplicar_correcoes(PPt, dias, Tmed, IAF=5, HI=0.35, U=0.15):
    """
    Retorna:
      - Y_final: produtividade potencial (kg/ha) ajustada para umidade final U
      - BMS: biomassa bruta (kg/ha)
      - PMS: biomassa seca (kg/ha) -> exibida como 'Biomassa (kg/ha)'
    """
    frac_resp = 0.6 if Tmed < 20 else 0.5
    CIAF = calc_CIAF(IAF)
    BMS = PPt * dias * CIAF
    PMS = BMS * (1 - frac_resp)
    Y_MS = PMS * HI
    Y_final = Y_MS / (1 - U)
    return {"Y_final": Y_final, "BMS": BMS, "PMS": PMS}

def _doy_to_month_day(doy):
    dias_mes=[31,28,31,30,31,30,31,31,30,31,30,31]; m=1
    for dm in dias_mes:
        if doy<=dm: return m,doy
        doy-=dm; m+=1
    return 12,31

def aplicar_reducao_hidrica(Yp, df_mensal, mes_inicio, dia_inicio, dias_ciclo, ky):
    """
    Ajusta a produtividade potencial (Yp) para produtividade atingível (Ya),
    com base na equação de Stewart/FAO:
        Ya = Yp * [1 - ky * (DEF_total / ETo_total)]
    O DEF_total e ETo_total são somados ao longo do ciclo,
    ponderando pelos dias de cada mês.
    """
    dias_mes_fix = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    DEF_total, ETo_total = 0.0, 0.0
    m, d, dias_rest = int(mes_inicio), int(dia_inicio), float(dias_ciclo)

    while dias_rest > 0:
        dias_mes = float(dias_mes_fix[m])
        dias_util = min(dias_rest, dias_mes - (d - 1))
        row = df_mensal[df_mensal["MESNUM"] == m].iloc[0]
        frac = dias_util / dias_mes
        DEF_total += float(row["DEF"]) * frac
        ETo_total += float(row["ETo"]) * frac

        dias_rest -= dias_util
        m = 1 if m == 12 else m + 1
        d = 1  # próximos meses começam no dia 1

    fator = 1.0
    if ETo_total > 0:
        fator = 1.0 - ky * (DEF_total / ETo_total)
    fator = max(0.0, fator)  # evita valores negativos

    Ya = Yp * fator
    reducao_pct = (1.0 - (Ya / Yp)) * 100.0 if Yp > 0 else 0.0
    return Ya, reducao_pct

def aplicar_reducao_fases_ETc(
    Yp, df_mensal, mes_inicio, dia_inicio, dias_ciclo,
    ky_veg, ky_rep, Kc_veg, Kc_rep, prop_rep
):
    """
    Redução por fases (vegetativa e reprodutiva) usando:
        ETc_fase = Kc_fase × ΣETo_ref_fase
        ETa_fase ≈ Kc_fase × ΣETR_ref_fase
    e Stewart faseado:
        1 - Ya/Yp = ky_veg*(1 - ETa_veg/ETc_veg) + ky_rep*(1 - ETa_rep/ETc_rep)

    Onde ΣETo_ref_fase e ΣETR_ref_fase são acumulados proporcionalmente aos dias do ciclo
    dentro de cada mês. A fase vegetativa vem primeiro; a reprodutiva depois.
    """
    dias_mes_fix = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

    # dividir o ciclo em fases
    dias_ciclo = float(dias_ciclo)
    dias_rep = dias_ciclo * float(prop_rep)
    dias_veg = max(0.0, dias_ciclo - dias_rep)

    # acumuladores
    ETo_veg_ref = 0.0
    ETR_veg_ref = 0.0
    ETo_rep_ref = 0.0
    ETR_rep_ref = 0.0

    # ponteiros
    m = int(mes_inicio)
    d = int(dia_inicio)
    dias_rest = float(dias_ciclo)
    veg_rest = float(dias_veg)

    while dias_rest > 0:
        dias_mes = float(dias_mes_fix[m])
        dias_util = min(dias_rest, dias_mes - (d - 1))

        # fração do mês usado
        frac_mes = dias_util / dias_mes

        # valores do mês no df_mensal
        row = df_mensal[df_mensal["MESNUM"] == m].iloc[0]
        ETo_seg_total = float(row["ETo"]) * frac_mes
        ETR_seg_total = float(row["ETR"]) * frac_mes

        # repartir entre fases
        dias_veg_usados = min(dias_util, max(0.0, veg_rest))
        dias_rep_usados = dias_util - dias_veg_usados

        if dias_util > 0:
            frac_veg_no_seg = dias_veg_usados / dias_util
            frac_rep_no_seg = dias_rep_usados / dias_util
        else:
            frac_veg_no_seg = frac_rep_no_seg = 0.0

        # acumular na vegetativa
        ETo_veg_ref += ETo_seg_total * frac_veg_no_seg
        ETR_veg_ref += ETR_seg_total * frac_veg_no_seg

        # acumular na reprodutiva
        ETo_rep_ref += ETo_seg_total * frac_rep_no_seg
        ETR_rep_ref += ETR_seg_total * frac_rep_no_seg

        # avançar
        veg_rest -= dias_veg_usados
        dias_rest -= dias_util
        m = 1 if m == 12 else m + 1
        d = 1

    # aplicar Kc em cada fase
    ETc_veg = Kc_veg * ETo_veg_ref
    ETa_veg = Kc_veg * ETR_veg_ref
    ETc_rep = Kc_rep * ETo_rep_ref
    ETa_rep = Kc_rep * ETR_rep_ref

    # déficits relativos
    red_veg = 0.0 if ETc_veg <= 0 else (1.0 - (ETa_veg / ETc_veg))
    red_rep = 0.0 if ETc_rep <= 0 else (1.0 - (ETa_rep / ETc_rep))

    # perda total (somar fases)
    perda_rel = ky_veg * red_veg + ky_rep * red_rep
    perda_rel = max(0.0, min(1.0, perda_rel))  # truncar em [0,1]

    Ya = float(Yp) * (1.0 - perda_rel)
    reducao_pct = perda_rel * 100.0

    return Ya, reducao_pct



def simular_52_epocas(df,Tb,GD_alvo,grupo,IAF=5,HI=0.45,U=0.15):
    linhas=[]
    for wk in range(52):
        doy=1+7*wk
        mes,dia=_doy_to_month_day(doy)
        medias=medias_ciclo_termico(df,mes,dia,Tb,GD_alvo)
        res_ppb=ppb_fao_diaria(medias["Q0"],medias["Qg"],medias["Tmed"],medias["nN"],grupo)
        res_corr=aplicar_correcoes(res_ppb["PPt"],medias["dias_total"],medias["Tmed"],IAF,HI,U)
        linhas.append({"Semana":wk+1,"Inicio":f"{dia:02d}/{mes:02d}","Prod_final":res_corr["Y_final"]})
    return pd.DataFrame(linhas)

# ------------------------------------------------------------
# BLOCO 4: Interface
# ------------------------------------------------------------
st.title(
    "🌱 Programa para estimativa da produtividade potencial e atingível "
    "pelo método Zona Agroecológica (FAO)\n"
    "Desenvolvido pelo Prof. Cláudio Ricardo da Silva em 20/09/2025 - Universidade Federal de Uberlândia"
)

col1,col2,col3=st.columns([2,2,1])
with col1:
    lat=st.number_input("**Latitude**",-60.0,60.0,-18.91,step=0.01)
    lon=st.number_input("**Longitude**",-90.0,-30.0,-48.26,step=0.01)
    grupo=st.selectbox(
        "**Grupo de cultura**",
        [1,2,3],
        format_func=lambda x: {
            1: "Leguminosas de inverno (C3 inverno)",
            2: "Leguminosas de verão (C3 verão)",
            3: "Gramíneas (C4)"
        }[x]
    )
     # ---- Proporção do ciclo reprodutivo ----
    prop_rep = st.number_input(
        "**Proporção reprodutiva do ciclo (%)**",
        min_value=10, max_value=70, value=40, step=5
    ) / 100.0

    
    # ---- Coeficientes de cultivo (Kc) por fases ----
    Kc_sug = {
        1: (0.70, 1.05),  # C3 inverno
        2: (0.75, 1.10),  # C3 verão
        3: (0.80, 1.15),  # C4
    }
    Kc_veg_def, Kc_rep_def = Kc_sug.get(grupo, (0.80, 1.10))   
     
    Kc_veg = st.number_input("**Coeficiente de cultivo - Kc (fase vegetativa)**", 
                             0.4, 1.5, float(Kc_veg_def), step=0.05)
    Kc_rep = st.number_input("**Coeficiente de cultivo - Kc (fase reprodutiva)**", 
                             0.4, 1.5, float(Kc_rep_def), step=0.05)
   

with col2:
    data=st.date_input("**Data de semeadura**",value=date(2025,9,12))
    mes_inicio,dia_inicio=data.month,data.day
    Tb=st.number_input("**Temperatura base (°C)**",0,20,10,step=1)
    GD_alvo=st.number_input("**Soma térmica (°C.d)**",500,4000,1300,step=5)
    CAD = st.number_input("**Capacidade de água disponível (CAD, mm)**", 50, 250, 100, step=10)
    eto_metodo = st.selectbox(
        "**Método para ETo**",
        ["Penman simplificado", "Thornthwaite"],
        index=0
    )
with col3:
    HI=st.slider("**Índice de colheita (HI)**",0.1,0.9,0.45,step=0.01)
    U=st.slider("**Umidade final**",0.05,0.3,0.15,step=0.01)
    IAF_max=st.slider("**IAF máximo**",1.0,5.0,5.0,step=0.1)
    # ---- Coeficientes de sensibilidade (Ky) por fases ----
    ky_sug = {
        1: (0.80, 1.10),  # C3 inverno
        2: (0.90, 1.00),  # C3 verão
        3: (1.00, 1.30),  # C4
    }
    ky_veg_def, ky_rep_def = ky_sug.get(grupo, (0.90, 1.10))

    ky_veg = st.slider("**Coeficiente de sensibilidade - Ky (fase vegetativa)**", 
                       0.5, 1.8, float(ky_veg_def), step=0.05)
    ky_rep = st.slider("**Coeficiente de sensibilidade - Ky (fase reprodutiva)**", 
                       0.5, 1.8, float(ky_rep_def), step=0.05)



# ------------------------------------------------------------
# Helper para esconder o índice do DataFrame de forma compatível
# ------------------------------------------------------------
def _styler_hide_index_compat(styler):
    try:
        return styler.hide(axis="index")   # pandas >= 1.4
    except Exception:
        try:
            return styler.hide_index()     # pandas mais antigas
        except Exception:
            return styler

# ------------------------------------------------------------
# Rodar simulação
# ------------------------------------------------------------
# # ------------------------------------------------------------
# Rodar simulação (simples: tenta NASA → se falhar, avisa e para)
# ------------------------------------------------------------
if st.button("🚀 Rodar simulação"):
    # 1) Teste rápido de DNS (mensagem clara e para)
    try:
        socket.gethostbyname("power.larc.nasa.gov")
    except Exception as e:
        st.error(f"Não foi possível resolver power.larc.nasa.gov (DNS). Detalhes: {e}")
        st.info("Verifique seu Wi-Fi/DNS (ex.: 8.8.8.8/1.1.1.1) e tente novamente.")
        st.stop()

    # 2) Baixar dados da NASA
    try:
        df_mensal = baixar_dados(lat, lon, eto_metodo)
    except RuntimeError as e:
        st.error("Não consegui baixar os dados da NASA POWER agora.")
        st.info("Conexão ok, mas o serviço pode estar instável. Tente novamente mais tarde.")
        st.caption(f"Detalhes técnicos: {e}")
        st.stop()

    # 3) Balanço hídrico
    df_mensal = balanco_hidrico(df_mensal, CAD)

    # 4) Abas (uma única vez)
    aba1, aba2 = st.tabs(["🌾 Produtividade FAO/AEZ", "☀️ Clima + Balanço Hídrico"])

    # =========================
    # ABA 1: Produtividade
    # =========================
    with aba1:
        # cálculos do ciclo
        medias = medias_ciclo_termico(df_mensal, mes_inicio, dia_inicio, Tb, GD_alvo)
        res_ppb = ppb_fao_diaria(
            medias["Q0"], medias["Qg"], medias["Tmed"], medias["nN"], grupo
        )
        res_corr = aplicar_correcoes(
            res_ppb["PPt"], medias["dias_total"], medias["Tmed"], IAF_max, HI, U
        )

        # --- Produtividade atingível (usando Ky e Kc por fases) ---
        Yp = float(res_corr["Y_final"])  # produtividade potencial do ciclo
        Ya, reducao_pct = aplicar_reducao_fases_ETc(
        Yp, df_mensal, mes_inicio, dia_inicio, medias["dias_total"],
        ky_veg, ky_rep, Kc_veg, Kc_rep, 0.40   # por enquanto fixamos 40% rep
        )
  
        # (a) Tabela 1: Dados médios (1 casa, alinhado)
        st.subheader("📊 Dados médios (ciclo)")
        dados_medios_df = pd.DataFrame([{
            "Tmed (°C)": medias["Tmed"],
            "Q0 (MJ/m²·d)": medias["Q0"],
            "Qg (MJ/m²·d)": medias["Qg"],
            "n/N": medias["nN"],
        }])
        dados_medios_styler = (
            dados_medios_df.round(1)
            .style
            .format("{:.1f}")
            .set_table_styles([
                {"selector": "th", "props": "text-align:center; font-weight:bold;"},
                {"selector": "td", "props": "text-align:right;"},
            ])
        )
        st.table(_styler_hide_index_compat(dados_medios_styler))

                # (b) Tabela 2: Índices do ciclo (inclui potencial x atingível)
        st.subheader("📌 Índices do ciclo")
        indices_df = pd.DataFrame([{
            "CTn": res_ppb["CTn"],
            "CTc": res_ppb["CTc"],
            "Ciclo (dias)": medias["dias_total"],
            "Biomassa (kg/ha)": res_corr["PMS"],
            "Produtividade potencial (kg/ha)": Yp,
            "Produtividade atingível (kg/ha)": Ya,
            "Redução por déficit (%)": reducao_pct,
        }])

        indices_styler = (
            indices_df.round(1)
            .style
            .format({
                "CTn": "{:.2f}",
                "CTc": "{:.2f}",
                "Ciclo (dias)": "{:.0f}",
                "Biomassa (kg/ha)": "{:.0f}",
                "Produtividade potencial (kg/ha)": "{:.0f}",
                "Produtividade atingível (kg/ha)": "{:.0f}",
                "Redução por déficit (%)": "{:.1f}%"
            })
            .set_table_styles([
                {"selector": "th", "props": "text-align:center; font-weight:bold;"},
                {"selector": "td", "props": "text-align:right;"},
            ])
            .set_properties(
                subset=["Produtividade potencial (kg/ha)"],
                **{"color": "red", "font-weight": "bold"}
            )
            .set_properties(
                subset=["Produtividade atingível (kg/ha)"],
                **{"color": "blue", "font-weight": "bold"}
            )
        )
        st.table(_styler_hide_index_compat(indices_styler))

                # gráfico: produtividade potencial x atingível por mês de semeadura
        df_52 = simular_52_epocas(df_mensal, Tb, GD_alvo, grupo, IAF_max, HI, U)
        meses_ordem = {
            1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
        }
        df_52["MesNum"] = df_52["Inicio"].str[3:5].astype(int)
        df_52["Mes"] = df_52["MesNum"].map(meses_ordem)

        # produtividade potencial média por mês
        df_mes_pot = (
            df_52.groupby("Mes", sort=False)["Prod_final"]
            .mean()
            .reindex(list(meses_ordem.values()))
        )
        # produtividade atingível média por mês
        df_mes_at = []
        for mes in range(1, 13):
            sub = df_52[df_52["MesNum"] == mes]
            if not sub.empty:
                Yp_med = sub["Prod_final"].mean()
                Ya_med, _ = aplicar_reducao_fases_ETc(
                    Yp_med, df_mensal, mes, 1, medias["dias_total"],
                    ky_veg, ky_rep, Kc_veg, Kc_rep, prop_rep
            )
                df_mes_at.append(Ya_med)
            else:
                df_mes_at.append(np.nan)

        # gráfico de barras lado a lado
        fig, ax = plt.subplots()
        x = np.arange(len(df_mes_pot))
        largura = 0.35
        ax.bar(x - largura/2, df_mes_pot.values, largura, color="red", alpha=0.7, label="Potencial")
        ax.bar(x + largura/2, df_mes_at, largura, color="blue", alpha=0.7, label="Atingível")
        ax.set_xticks(x)
        ax.set_xticklabels(df_mes_pot.index)
        ax.set_ylabel("kg/ha")
        ax.set_title("Produtividade Potencial x Atingível por mês de semeadura")
        ax.legend()
        st.pyplot(fig)



    # =========================
    # ABA 2: Clima + BHC
    # =========================
    with aba2:
        st.subheader("☀️ Dados climáticos + BHC (mensal)")
        st.dataframe(df_mensal.round(2))

        resumo = {
            "P_total (mm)": df_mensal["P_total"].sum(),
            "ETo (mm)":     df_mensal["ETo"].sum(),
            "ETR (mm)":     df_mensal["ETR"].sum(),
            "DEF (mm)":     df_mensal["DEF"].sum(),
            "EXC (mm)":     df_mensal["EXC"].sum()
        }
        st.table(pd.DataFrame([resumo]).round(1))

        st.subheader("📊 Extrato mensal do BHC")
        x = np.arange(1, len(df_mensal) + 1, dtype=float)
        labels = df_mensal["MES"].tolist()
        exc = pd.to_numeric(df_mensal["EXC"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        defv = pd.to_numeric(df_mensal["DEF"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        fig5, ax5 = plt.subplots()
        ax5.fill_between(x, 0.0, exc,   alpha=0.7, label="EXC")
        ax5.fill_between(x, 0.0, -defv, alpha=0.7, label="DEF")
        ax5.axhline(0, linewidth=0.8)
        ax5.set_ylabel("mm")
        ax5.set_title("Extrato mensal do BHC")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels)
        ax5.legend(loc="upper right")
        st.pyplot(fig5)

        # download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_mensal.round(2).to_excel(writer, sheet_name="Clima_BHC", index=False)
            pd.DataFrame([resumo]).round(1).to_excel(writer, sheet_name="Resumo", index=False)
        st.download_button(
            "⬇️ Baixar clima+BHC (XLSX)",
            buffer.getvalue(),
            "clima_bhc.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
