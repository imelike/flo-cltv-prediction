##############################################################
# FLO: BG-NBD ve Gamma-Gamma ile CLTV Tahmini
##############################################################

# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete
# sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online
# hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden
# oluşmaktadır.

#####################################
# Features
#####################################
# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from helpers.eda import check_df
from helpers.data_prep import missing_values_table

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

##############################################################
# Görev 1: Veriyi Anlama ve Hazırlama
##############################################################

# Step 1: flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

check_df(df)
df.describe().T

# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve
# replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
df.describe().T
replace_with_thresholds(df,"customer_value_total_ever_online")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"order_num_total_ever_online")


# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

# For omnichannel total over {online + offline}  (toplam satın alma sayısı)
df["total_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Total cost for omnichannel (toplam harcama tutarı)
df["total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz
df.dtypes
convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[convert] = df[convert].apply(pd.to_datetime)


##############################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
##############################################################

# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız
last_date = df["last_order_date"].max()  #2021-05-30
today_date = dt.datetime(2021, 6, 2)

# Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin
# yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak,
# recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame({"CUSTOMER_ID": df["master_id"],
             "RECENCY_CLTV_WEEKLY": (df["last_order_date"] - df["first_order_date"]).dt.days / 7,
             "T_WEEKLY": ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "FREQUENCY": df["total_of_purchases"],
             "MONETARY_CLTV_AVG": df["total_expenditure"] / df["total_of_purchases"]})

cltv_df.head()

cltv_df.info()


##############################################################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
##############################################################

# Adım 1: BG/NBD modelini fit ediniz
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['FREQUENCY'],
        cltv_df['RECENCY_CLTV_WEEKLY'],
        cltv_df['T_WEEKLY'])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.
cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                              cltv_df['FREQUENCY'],
                                              cltv_df['RECENCY_CLTV_WEEKLY'],
                                              cltv_df['T_WEEKLY'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.
cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                              cltv_df['FREQUENCY'],
                                              cltv_df['RECENCY_CLTV_WEEKLY'],
                                              cltv_df['T_WEEKLY'])


# Adım 2: Gamma-Gamma modelini fit ediniz.
# Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['FREQUENCY'], cltv_df['MONETARY_CLTV_AVG'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['FREQUENCY'], cltv_df['MONETARY_CLTV_AVG'])


# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz
# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['FREQUENCY'],
                                   cltv_df['RECENCY_CLTV_WEEKLY'],
                                   cltv_df['T_WEEKLY'],
                                   cltv_df['MONETARY_CLTV_AVG'],
                                   time=6,    # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi. (weekly, haftalık cins)
                                   discount_rate=0.01)

cltv.head()
cltv_df.head()

cltv_df["CLTV"] = cltv

cltv_df.sort_values(by="CLTV", ascending=False).head(20)


##############################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
##############################################################

# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba(segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_df["segment"] = pd.qcut(cltv_df["CLTV"], 4, labels=["D", "C", "B", "A"])
cltv_df.sort_values(by="CLTV", ascending=False).head(10)


# Adım 2:  4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
cltv_df.groupby("segment").agg({"count", "mean", "sum"})
