# LightFM Recommendation System

This repository contains a collaborative filtering recommendation system using the **LightFM** library. The model is trained on user-item interaction data and provides personalized recommendations.

## 📌 Features
- Uses **LightFM** for hybrid collaborative filtering.
- Provides **personalized recommendations** for users.
- Uses **implicit feedback data** to learn user preferences.

## 🛠 Installation
To use this project, follow these steps:

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/lightfm-recommendation.git
   cd lightfm-recommendation
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   Ensure that **LightFM** is installed:
   ```sh
   pip install lightfm
   ```

## 🚀 Usage
### **Training the Model**
```python
from lightfm import LightFM
import numpy as np

# Train the model
model = LightFM(loss='warp')  # WARP optimizes for ranking
model.fit(data['train'], epochs=30, num_threads=2)
```

### **Generating Recommendations**
```python
def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        
        print("User %s" % user_id)
        print("    Known positives:")
        for x in known_positives[:3]:
            print("        %s" % x)
        
        print("    Recommended:")
        for x in top_items[:3]:
            print("        %s" % x)

# Example usage
sample_recommendation(model, data, [3, 25, 450])
```

## 📊 Dataset
Make sure your `data` dictionary contains:
- `data['train']`: A sparse matrix with user-item interactions.
- `data['item_labels']`: A list of item names corresponding to indices.

## 🔗 Contributing
Feel free to contribute by opening an **issue** or a **pull request**!

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💡 Acknowledgments
- **[LightFM Documentation](https://making.lyst.com/lightfm/docs/)** for their awesome library.
- Inspiration from various recommendation system research papers.

---

🚀 **Happy coding!**

