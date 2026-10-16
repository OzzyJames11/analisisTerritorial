import pandas as pd
import json

print("Iniciando auditoría de descartes...")

# Cargar el Excel
df = pd.read_excel("data/oferta_academica.xlsx")

# Cargar la configuración
with open("data/carreras_interes.json", "r", encoding="utf-8") as f:
    lista_interes = json.load(f)

carreras_validas = [c.strip().lower() for c in lista_interes]

def es_registro_valido(row):
    nombre = str(row.get('nombre', '')).strip().lower()
    homologa = str(row.get('epn_homologa', '')).strip().lower()
    
    if homologa == "no registra" or homologa == "nan":
        if nombre not in carreras_validas:
            return False
            
    if nombre in carreras_validas:
        return True
    if homologa in carreras_validas:
        return True
        
    return False

# Filtrar
mascara_validos = df.apply(es_registro_valido, axis=1)
df_descartados = df[~mascara_validos].copy()

print(f"Total de registros a descartar: {len(df_descartados)}")
print(f"Total de registros a subir a MongoDB: {len(df[mascara_validos])}")

# Exportar para revisión manual
df_descartados.to_csv("outputs/lista_descartados.csv", index=False)
print("\nSe ha exportado el detalle de los descartados a: outputs/lista_descartados.csv")
print("Revisa este archivo para asegurarte de que ninguna carrera útil se haya quedado fuera.")
