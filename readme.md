# 🧠 LLM-Driven Arithmetic Synthesis for In-Memory Computing

This project reimagines the core ideas from the paper **"Automated Synthesis for In-Memory Computing (AUTO)"** by implementing a complete in-memory arithmetic synthesis engine. 

Instead of using a traditional greedy algorithm to collapse the multiplication pyramid, this version uses **ChatGPT (an LLM)** to intelligently select the best-fitting custom adders.

---

## 🚀 Key Features

- **LLM-Based Adder Selection**  
  Uses OpenAI GPT to choose the most cost-efficient custom adder at each step, guided by a scoring formula:  
  `S = (1s covered) / cost`.

- **Custom Adder Library**  
  Adders are defined in `lookuptable.csv`. Each entry includes:
  - 5 columns for shape (heights)
  - 9th column for cost

- **Fallback Mechanism with Vector Similarity**  
  When GPT cannot produce a valid move, the system falls back to a similarity-matching method in `closest_adder.py`.

- **Full Pyramid Simulator**  
  Handles partial product formation, adder stamping, carry bit generation, gravity simulation, and result compression.

- **Interactive Flask Web App**  
  Allows full pyramid initialization, adder simulation, and GPT-driven synthesis from the browser.

---

## 📁 Project Structure

```
.
├── .env                    # API key (OPENAI_API_KEY)
├── .gitignore              # Ignore env, cache, etc.
├── closest_adder.py        # Fallback mechanism using vector matching
├── info.txt                # Notes or config info
├── lookuptable.csv         # Custom adders (shape + cost)
├── main_app.py             # Full logic + Flask backend
├── templates/
│   └── dashboard.html      # (Optional) Frontend interface
└── readme.md               # You're reading it!
```

---

## 🧠 How It Works

1. **Pyramid Generation**  
   Binary multiplication is modeled as a pyramid of partial products (`1`s), ending in an `X`.

2. **GPT-Guided Adder Selection**  
   Each step:
   - Finds all valid adders and placements
   - Scores them with `S = coverage / cost`
   - Asks GPT to choose the best adder and position
   - Validates and applies the choice

3. **Carry Bit Calculation**  
   Determines how many carry bits (`*`) are needed for each adder and inserts them in the pyramid.

4. **Gravity Simulation**  
   Consumed bits disappear, and remaining bits fall straight down. The process continues.

5. **Fallback Using Closest Adder**  
   If GPT fails or makes an invalid choice, the system:
   - Extracts diagonal bit heights
   - Finds the closest matching shape using vector similarity
   - Applies that adder and resumes

6. **Final Collapse**  
   When only 1s remain, the system collapses them into a final `111...1X` row.

---

## 🌐 Running the App

1. **Install dependencies**:
   ```bash
   pip install flask openai
   ```

2. **Set your API key in `.env`**:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ```

3. **Run the app**:
   ```bash
   python main_app.py
   ```

4. **Access in browser**:
   ```
   http://localhost:5000/
   ```

---

## 🔌 API Endpoints

| Route                  | Method | Description                              |
|-----------------------|--------|------------------------------------------|
| `/`                   | GET    | UI Dashboard                             |
| `/init_pyramid`       | POST   | Create a new pyramid (N-bit)             |
| `/fetch_custom_adder` | GET    | Full GPT-based synthesis loop            |
| `/apply_adder`        | POST   | Manually apply a specific adder          |
| `/load_adders`        | GET    | Load custom adders from CSV              |
| `/logs`               | GET    | Retrieve log of actions                  |
| `/visualize`          | GET    | Return current pyramid as text           |

---

## 📊 Adder CSV Format

Example row in `lookuptable.csv`:

```
1,2,3,0,0,...,7
```

- First 5 numbers: shape (non-zero heights of columns, right to left)
- 9th value: cost of applying this adder

---

## 💡 Innovation & Contribution

- Replaces a greedy algorithm with a **GPT-driven decision engine**
- Builds an end-to-end **adaptive arithmetic optimizer**
- Demonstrates how **LLMs can reason over symbolic structures**
- Opens doors for **AI-guided hardware synthesis**

---

## 📜 License

MIT License — free to use, modify, and share. If helpful, please cite this repo.

---

## 🙌 Acknowledgments

- Inspired by: *"Automated Synthesis for In-Memory Computing"* (ICCAD 2023)
- Built using: OpenAI GPT, Flask, and your engineering brilliance!
