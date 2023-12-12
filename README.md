# AI
Internship task &amp; projects on AI
///CHATBOT WITH RULE-BASED
import re

def simple_chatbot(user_input):
    # Define rules and responses
    rules = {
        r'hello|hi|hey': 'Hello! How can I help you?',
        r'how are you': 'I am just a computer program, but thanks for asking!',
        r'what is your name': 'I am a simple chatbot.',
        r'bye|goodbye': 'Goodbye! Have a great day!',
        r'(\d+) (plus|minus|times|divided by) (\d+)': calculate_math,
        r'(.*)': 'I am not sure how to respond to that. Can you be more specific?',
    }

    # Check for matches with rules
    for pattern, response in rules.items():
        if re.match(pattern, user_input, re.IGNORECASE):
            if callable(response):
                return response(user_input)
            else:
                return response

def calculate_math(user_input):
    match = re.match(r'(\d+) (plus|minus|times|divided by) (\d+)', user_input, re.IGNORECASE)
    if match:
        num1, operation, num2 = match.groups()
        num1, num2 = int(num1), int(num2)
        if operation.lower() == 'plus':
            return f'The result is: {num1 + num2}'
        elif operation.lower() == 'minus':
            return f'The result is: {num1 - num2}'
        elif operation.lower() == 'times':
            return f'The result is: {num1 * num2}'
        elif operation.lower() == 'divided by':
            if num2 == 0:
                return 'Cannot divide by zero!'
            else:
                return f'The result is: {num1 / num2}'
    else:
        return 'Invalid math expression.'

# Main loop for the chatbot
while True:
    user_input = input('You: ')
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print('Chatbot: Goodbye!')
        break
    response = simple_chatbot(user_input)
    print('Chatbot:', response)

///TIC-TAC-TOE AI    
import copy

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def is_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for row in board:
        if all(cell == player for cell in row):
            return True

    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True

    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

def is_draw(board):
    # Check if the board is full (draw)
    return all(board[row][col] != ' ' for row in range(3) for col in range(3))

def get_empty_cells(board):
    # Return a list of empty cells on the board
    return [(row, col) for row in range(3) for col in range(3) if board[row][col] == ' ']

def minimax(board, depth, maximizing_player):
    if is_winner(board, 'O'):
        return -1
    if is_winner(board, 'X'):
        return 1
    if is_draw(board):
        return 0

    if maximizing_player:
        max_eval = float('-inf')
        for row, col in get_empty_cells(board):
            board[row][col] = 'X'
            eval = minimax(board, depth + 1, False)
            board[row][col] = ' '
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for row, col in get_empty_cells(board):
            board[row][col] = 'O'
            eval = minimax(board, depth + 1, True)
            board[row][col] = ' '
            min_eval = min(min_eval, eval)
        return min_eval

def get_best_move(board):
    best_val = float('-inf')
    best_move = None

    for row, col in get_empty_cells(board):
        board[row][col] = 'X'
        move_val = minimax(board, 0, False)
        board[row][col] = ' '

        if move_val > best_val:
            best_move = (row, col)
            best_val = move_val

    return best_move

def main():
    board = [[' ' for _ in range(3)] for _ in range(3)]

    while True:
        print_board(board)

        # Player's move
        row, col = map(int, input("Enter your move (row and column, separated by space): ").split())
        if board[row][col] == ' ':
            board[row][col] = 'O'
        else:
            print("Invalid move. Cell already occupied. Try again.")
            continue

        # Check if the player wins
        if is_winner(board, 'O'):
            print_board(board)
            print("Congratulations! You win!")
            break

        # Check for a draw
        if is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        # AI's move
        print("AI's move:")
        ai_row, ai_col = get_best_move(board)
        board[ai_row][ai_col] = 'X'

        # Check if the AI wins
        if is_winner(board, 'X'):
            print_board(board)
            print("AI wins! Better luck next time.")
            break

        # Check for a draw
        if is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()

/////task 3 ==== IMAGE CAPTIONING
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load pre-trained ResNet model
class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, image):
        with torch.no_grad():
            features = self.resnet(image)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features

# Load pre-trained word embeddings
class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

# Transform input image to match ResNet requirements
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

# Generate a caption for a given image
def generate_caption(image_path, encoder, decoder, transform, max_length=20):
    image = load_image(image_path, transform)
    feature = encoder(image)
    sampled_ids = []
    inputs = feature.unsqueeze(1)

    for _ in range(max_length):
        outputs = decoder(feature, inputs)
        _, predicted = outputs.max(2)
        sampled_ids.append(predicted.item())
        inputs = decoder.embedding(predicted)

    return sampled_ids

# Main function
def main():
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load pre-trained models
    embed_size = 256
    hidden_size = 512
    vocab_size = 10000  # Adjust based on your dataset
    encoder = ImageEncoder(embed_size)
    decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)

    # Load weights (adjust the paths accordingly)
    encoder.load_state_dict(torch.load('encoder_weights.pth'))
    decoder.load_state_dict(torch.load('decoder_weights.pth'))

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    # Test image captioning
    image_path = 'path/to/your/image.jpg'
    captions = generate_caption(image_path, encoder, decoder, transform)

    # Display the generated caption
    print("Generated Caption:", captions)

if __name__ == "__main__":
    main()
////task 4 ===
#RECOMMENDATION SYSTEM
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load MovieLens dataset (adjust the path accordingly)
ratings_data = pd.read_csv('path/to/ratings.csv')
movies_data = pd.read_csv('path/to/movies.csv')

# Merge ratings and movies data
movie_ratings = pd.merge(ratings_data, movies_data, on='movieId')

# Create a user-item matrix
user_item_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0
user_item_matrix = user_item_matrix.fillna(0)

# Perform train-test split
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(train_data)

# Function to get movie recommendations
def get_movie_recommendations(user_preferences, user_similarity, n=5):
    similar_scores = user_similarity.dot(user_preferences) / user_similarity.sum(axis=1)
    recommendation_scores = user_item_matrix.sum(axis=0) * (1 - user_preferences)
    recommendation_scores *= similar_scores.reshape(-1, 1)
    recommended_movies = recommendation_scores.sum(axis=0).sort_values(ascending=False)
    top_recommendations = recommended_movies.head(n)
    return top_recommendations.index.tolist()

# Example: Recommend movies for a user (replace userId with the desired user)
user_id = 1
user_preferences = user_item_matrix.loc[user_id].values.reshape(1, -1)
recommended_movies = get_movie_recommendations(user_preferences, user_similarity, n=5)

# Display the recommended movies
print(f"Top 5 Movie Recommendations for User {user_id}:")
for movie_title in recommended_movies:
    print(movie_title)

///task 5 =====
# FACE DETECTION AND RECOGNITION
    import cv2
import face_recognition

# Load pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face recognition model (optional, requires face_recognition library)
known_faces = []
known_names = []

# Function to detect faces in an image using Haar cascade
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

# Function to recognize faces using face_recognition library
def recognize_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        face_names.append(name)

    return face_names

# Function to draw bounding boxes around detected faces
def draw_bounding_boxes(image, faces, names):
    for (x, y, w, h), name in zip(faces, names):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Main function for face detection and recognition
def main():
    # Load an image or video (replace 'input_image.jpg' with your file path)
    input_file = 'input_image.jpg'
    input_type = input_file.split('.')[-1].lower()

    if input_type in ['jpg', 'jpeg', 'png']:
        image = cv2.imread(input_file)
        faces = detect_faces_haar(image)
        names = recognize_faces(image)

        draw_bounding_boxes(image, faces, names)

        cv2.imshow('Face Detection and Recognition', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # For video input
        cap = cv2.VideoCapture(input_file)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            faces = detect_faces_haar(frame)
            names = recognize_faces(frame)

            draw_bounding_boxes(frame, faces, names)

            cv2.imshow('Face Detection and Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

