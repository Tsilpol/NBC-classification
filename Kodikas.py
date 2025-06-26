# === ΕΙΣΑΓΩΓΗ ΒΙΒΛΙΟΘΗΚΩΝ ===
import pandas as pd  # Διαχείριση πινάκων δεδομένων (DataFrames)
import numpy as np  # Αριθμητικοί υπολογισμοί με πίνακες
from collections import defaultdict  # Λεξικά με αυτόματες default τιμές
from math import exp, sqrt, pi, log  # Μαθηματικές συναρτήσεις
from sklearn.model_selection import StratifiedKFold  # Stratified cross-validation

# === Βήμα 1: Φόρτωση και Ανάλυση Δεδομένων ===
df = pd.read_csv("iris.csv")  # Φόρτωση του dataset από αρχείο CSV
features = df.columns[1:-1]  # Επιλογή όλων των χαρακτηριστικών (εξαιρείται πιθανό ID και η κλάση)
class_col = df.columns[-1]  # Το όνομα της στήλης της κλάσης
classes = df[class_col].unique()  # Όλες οι μοναδικές ετικέτες κλάσεων

# === Βήμα 2: Ανάλυση Τύπων Χαρακτηριστικών ===
feature_info = {}  # Θα αποθηκεύσει τον τύπο κάθε χαρακτηριστικού (Διακριτό ή Συνεχές)
for feature in features:
    series = df[feature]  # Η στήλη του χαρακτηριστικού
    if series.dtype == object or series.nunique() < 10:  # Αν είναι κατηγορικό ή έχει λίγες τιμές
        feature_info[feature] = {
            'type': 'D',  # Discrete
            'values': sorted(series.dropna().astype(str).unique())  # Λίστα πιθανών τιμών
        }
    else:  # Αλλιώς είναι Συνεχές
        feature_info[feature] = {
            'type': 'C',  # Continuous
            'range': (series.min(), series.max())  # Εύρος τιμών
        }

# Εκτύπωση περιγραφής χαρακτηριστικών
print("=== ΤΥΠΟΣ ΔΕΔΟΜΕΝΩΝ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ ===")
# Διάσχιση όλων των χαρακτηριστικών και των πληροφοριών τους από το λεξικό feature_info
for feat, info in feature_info.items():
    
    # Αν το χαρακτηριστικό είναι διακριτό (κατηγορικό)
    if info['type'] == 'D':
        # Εκτύπωσε το όνομα του χαρακτηριστικού και τις πιθανές κατηγορικές του τιμές
        print(f"{feat} (D): {info['values']}")
    
    else:
        # Αν είναι συνεχές χαρακτηριστικό, από το range παίρνουμε την ελάχιστη και μέγιστη τιμή
        lo, hi = info['range']
        # Εκτύπωσε το όνομα του χαρακτηριστικού και το εύρος των τιμών του, με 2 δεκαδικά ψηφία
        print(f"{feat} (C): εύρος {lo:.2f}–{hi:.2f}")

# Εκτύπωση της στήλης κλάσης και των δυνατών ετικετών (classes) που υπάρχουν στο dataset
print(f"Κλάση {class_col} (D): {sorted(classes)}")


# === Βήμα 3: Συνάρτηση Gaussian PDF ===
def gaussian_pdf(x, mean, var):
    if var == 0 or np.isnan(var):  # Αν η διασπορά είναι μηδέν ή NaN, δώσε μικρή τιμή
        var = 1e-6
    coeff = 1 / sqrt(2 * pi * var)  # Συντελεστής της Gaussian
    exponent = exp(-((x - mean) ** 2) / (2 * var))  # Εκθετικό μέρος
    return coeff * exponent  # Επιστροφή πυκνότητας πιθανότητας

# === Βήμα 4: Εκπαίδευση Naive Bayes Ταξινομητή ===
def train_naive_bayes(data):
    priors = data[class_col].value_counts(normalize=True).to_dict()  # A priori πιθανότητες
    gaussian_params = defaultdict(dict)  # Μέσος και διασπορά για συνεχόμενα
    discrete_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # Πιθανότητες για κατηγορικά

    for cls in classes:  # Για κάθε κλάση
        cls_data = data[data[class_col] == cls] # Φιλτράρει το DataFrame data και κρατά μόνο τις εγγραφές που ανήκουν στην κατηγορία cls
        for feat in features:
            if feature_info[feat]['type'] == 'C': # Αν το χαρακτηριστικό είναι συνεχές (Continuous), εφαρμόζουμε την κατανομή Gauss.
                mu = cls_data[feat].mean()  # Μέσος
                sigma2 = cls_data[feat].var()  # Διασπορά
                gaussian_params[cls][feat] = (mu, sigma2)
            else:
                values = feature_info[feat]['values']
                total = len(cls_data) # Υπολογίζουμε πόσα δείγματα υπάρχουν συνολικά στην τρέχουσα κλάση `cls`
                for val in values:
                    count = (cls_data[feat].astype(str) == val).sum()
                    prob = (count + 1) / (total + len(values))  # Laplace smoothing
                    discrete_probs[cls][feat][val] = prob

    return priors, gaussian_params, discrete_probs

# Εκπαίδευση σε ολόκληρο το σύνολο δεδομένων
priors_full, gaussian_params_full, discrete_probs_full = train_naive_bayes(df)

# Εμφάνιση a priori πιθανοτήτων
print("\n=== A PRIORI ΠΙΘΑΝΟΤΗΤΕΣ ===")
for cls, p in priors_full.items():
    print(f"P({cls}) = {p:.4f}")

# Εμφάνιση παραμέτρων για Gaussian χαρακτηριστικά
print("\n=== GAUSSIAN ΠΑΡΑΜΕΤΡΟΙ ===")
for cls in classes:  # Για κάθε κλάση στο σύνολο των κλάσεων (π.χ. 'yes', 'no' ή 0, 1)
    
    # Εκτύπωση τίτλου για την τρέχουσα κλάση
    print(f"Κλάση: {cls}")

    for feat in features:  # Για κάθε χαρακτηριστικό (στήλη) του dataset
        
        # Αν το χαρακτηριστικό είναι συνεχές (C), δηλαδή αριθμητικό
        if feature_info[feat]['type'] == 'C':
            
            # Παίρνουμε τις αποθηκευμένες παραμέτρους Gaussian (μέσο και διασπορά)
            mu, sigma2 = gaussian_params_full[cls][feat]

            # Εκτύπωση των παραμέτρων για το συγκεκριμένο χαρακτηριστικό και κλάση
            print(f"  {feat}: μέσος={mu:.4f}, διασπορά={sigma2:.4f}")

    # Κενή γραμμή για καλύτερη αναγνωσιμότητα στην έξοδο μεταξύ των κλάσεων
    print()


# === Βήμα 5: Συνάρτηση Ταξινόμησης ===
def classify(sample, priors, gaussian_params, discrete_probs):
    log_post = {}
    for cls in classes:
        log_prob = log(priors[cls])  # Ξεκινάμε με το log(a priori)
        for i, feat in enumerate(features):
            if feature_info[feat]['type'] == 'C':
                mu, sigma2 = gaussian_params[cls][feat]
                p = gaussian_pdf(sample[i], mu, sigma2)
            else:
                val = str(sample[i])
                values = feature_info[feat]['values']
                p = discrete_probs[cls][feat].get(val, 1 / (len(values) + 1))
            p = max(p, 1e-300)  # Αποφυγή log(0)
            log_prob += log(p)
        log_post[cls] = log_prob

    max_log = max(log_post.values())  # Για αριθμητική σταθερότητα
    exp_scores = {cls: exp(log_post[cls] - max_log) for cls in classes}
    total = sum(exp_scores.values())
    return {cls: score / total for cls, score in exp_scores.items()}  # Κανονικοποίηση


# === Βήμα 6: Cross-Validation Αξιολόγηση ===
X = df[features]
y = df[class_col]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-πλή cross-validation με διατήρηση κατανομής
accuracies = []

for train_idx, test_idx in skf.split(X, y):  
    # Για κάθε fold του Stratified K-Fold cross-validation
    # Η συνάρτηση skf.split επιστρέφει δύο λίστες από δείκτες:# - train_idx: δείκτες για τα δεδομένα εκπαίδευσης # - test_idx: δείκτες για τα δεδομένα ελέγχου (validation)

    # Δημιουργούμε το υποσύνολο εκπαίδευσης με βάση τους δείκτες
    train_df = df.iloc[train_idx].reset_index(drop=True)
    
    # Δημιουργούμε το υποσύνολο ελέγχου (test set) με βάση τους δείκτες
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Εκπαιδεύουμε το Naive Bayes μοντέλο στο train_df:
    priors, gaussian_params, discrete_probs = train_naive_bayes(train_df)

    correct = 0 # Μετρητής σωστών προβλέψεων
    for i in range(len(test_df)): # Για κάθε δείγμα στο σύνολο ελέγχου (test set)
        xi = [test_df[feat].iloc[i] for feat in features]  # Δημιουργούμε μια λίστα με τις τιμές όλων των χαρακτηριστικών για το i-οστό δείγμα
        yi = test_df[class_col].iloc[i]  # Πραγματική κλάση του δείγματος (τιμή-στόχος για σύγκριση)
        probs = classify(xi, priors, gaussian_params, discrete_probs)  # Καλούμε τη συνάρτηση ταξινόμησης για να υπολογίσουμε τις πιθανότητες P(κλάση|x)
        pred = max(probs, key=probs.get)  # Πρόβλεψη = κλάση με μεγαλύτερη πιθανότητα
        if pred == yi: # Αν η πρόβλεψη είναι σωστή, αυξάνουμε τον μετρητή σωστών προβλέψεων
            correct += 1

    accuracy = correct / len(test_df) # Υπολογισμός ακρίβειας για το συγκεκριμένο fold: σωστά / σύνολο παραδειγμάτων
    accuracies.append(accuracy) # Προσθήκη της ακρίβειας αυτού του fold στη λίστα για συνολική αξιολόγηση στο τέλος

# Εκτύπωση αποτελεσμάτων αξιολόγησης
print("\n=== ΑΚΡΙΒΕΙΑ CROSS-VALIDATION ===")
print(f"Ακρίβεια ανά fold: {accuracies}")
print(f"Μέση ακρίβεια: {np.mean(accuracies):.4f}")

# === Βήμα 7: Εισαγωγή και Ταξινόμηση Νέου Δείγματος ===
print("\n=== ΕΙΣΑΓΩΓΗ ΚΑΙ ΤΑΞΙΝΟΜΗΣΗ ΔΕΙΓΜΑΤΟΣ ===")
print(f"Δώσε {len(features)} τιμές ({', '.join(features)}) χωρισμένες με κενό ή κόμμα: ")

try:
    raw = input("π.χ. 5.9 3.0 5.1 1.8 ")  # Εισαγωγή τιμών χρήστη
    vals = raw.replace(",", ".").split()  # Αντικατάσταση κόμμα με τελεία και split

    # Ανά τύπο χαρακτηριστικού, μετατροπή σε float ή διατήρηση κατηγορίας
    sample = [float(v) if feature_info[feat]['type'] == 'C' else v 
              for v, feat in zip(vals, features)]

    # Ταξινόμηση του δείγματος με βάση τις παραμέτρους του πλήρους συνόλου
    result = classify(sample, priors_full, gaussian_params_full, discrete_probs_full)

    # Εμφάνιση πιθανοτήτων για κάθε κλάση
    print("\n=== A POSTERIORI ΠΙΘΑΝΟΤΗΤΕΣ ===")
    for cls, p in result.items():
        print(f"P({cls}|x) = {p:.4f}")

except Exception as e:
    print("Σφάλμα:", e)  # Αν εισαχθούν λάθος τιμές, εμφάνιση σφάλματος