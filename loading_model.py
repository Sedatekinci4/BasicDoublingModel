import tensorflow as tf

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('C:\\Users\\tades\\PycharmProjects\\ModelforDuplicate\\duplicate_model.pb')

# Show the model architecture
new_model.summary()

num = int(input("Please enter an Integer: "))
# Try to predict the value given

res = new_model.predict([num])
print("\nOur models presume is: " + str(res))
print(res)
