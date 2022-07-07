import glob
import keras as kr
import numpy as np
import DATA.train as tr
import DATA.Tools as tls
import DATA.models as md
import matplotlib.pyplot as plt

# On crée le model
model = md.Bicephale(([64] * 4))
inputs = kr.Input(shape=(39), name="input")
x = model(inputs)
model = kr.Model(inputs, x, name="bicephale")

print(model.summary())
kr.utils.plot_model(
    model, "graph_model.png", show_shapes=True, show_layer_activations=True
)
model.save("bice_model")

# On divise notre model en ces deux sous parties
cla, est = md.cephalise(model)
cla.compile(optimizer=kr.optimizers.Adam(learning_rate=10e-4), loss="mse")
est.compile(optimizer=kr.optimizers.Adam(learning_rate=10e-4), loss="mse")

cutoff = 1e-6


pf = tls.pa.parquet.ParquetFile("Hey.parquet")
index = [pf.schema[i].name for i in range(4, 44)]


def seuil(df):
    return 1 * (np.abs(df["Int2e"]) > cutoff)


def xycla(df):
    return df[index[1:]], seuil(df)


def xyest(df):
    return df[index[1:]], df["Int2e"]


def split(df):
    x = np.abs(df["Int2e"])
    return np.sqrt(np.log(x + 1) / np.log(1 + cutoff)).astype("int64")


stats = tls.stats(pf)
Nf = max(split(stats.min), split(stats.max)) + 1
files = [f"spfiles/bi_{i}.parquet" for i in range(Nf)]


def goto(dtf):
    label = np.arange(Nf)
    return label[list(split(dtf))]


tls.sparq(pf, goto, "spfiles/bi", index, 10**5)
files = glob.glob("spfiles/bi_*.parquet")
prop = [tls.stats(files[i]).N / stats.N for i in range(len(files))]
prop = np.minimum(np.array(prop), 10 * np.min(prop))
prop[0] = 20 * np.sum(prop[1:])
print(prop)

comkwa = {"trainsize": 10**4, "sample": 10**5, "Nepoch": 5, "Nsets": 10}
x, y = xyest(tls.getrand(pf, 1400000, index, 10**5))
plt.subplot(131)
plt.plot(y, model(x.values), ".", markersize=1)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
plt.subplot(132)
plt.plot(y, cla(x.values), ".", markersize=1)
plt.subplot(133)
plt.plot(y, est(x.values), ".", markersize=1)
plt.savefig("etape_0.png", dpi=600)
plt.cla()
plt.clf()
plt.close()
for i in range(100):
    comkwa["nfile"] = "cla.csv"
    tr.train(cla, xycla, files, prop, **comkwa)
    comkwa["nfile"] = "est.csv"
    tr.train(est, xyest, files[1:], prop[1:], **comkwa)

    x, y = xyest(tls.getrand(pf, 1400000, index, 10**5))
    plt.subplot(131)
    plt.plot(y, model(x.values), ".", markersize=1)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    plt.subplot(132)
    plt.plot(y, cla(x.values), ".", markersize=1)
    plt.subplot(133)
    plt.plot(y, est(x.values), ".", markersize=1)
    plt.savefig(f"etape_{i+1}.png", dpi=600)
    plt.cla()
    plt.clf()
    plt.close()

    model.save("good_bi")
