import tensorflow as tf
from tensorflow import keras

# training only done on playing first
# grid values;
#   empty = -1
#   opponent = 1
#   player/model = 0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(3,3,)),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(9)                             
])

model.load_weights("weights.h5")
grid=[[-1 for i in range(3)] for j in range(3)]
print("successfully loaded neural net!")

# get Action Valuue
def getAV():
    state = tf.convert_to_tensor(grid)
    state = tf.expand_dims(state,0)

    return model(state)
    
def cpu_step():
    i = random.randrange(3)
    j = random.randrange(3)
    if grid[i][j]!=-1:
        cpu_step()
    else:
        grid[i][j] = 1
    return check()

def reset_grid():
  for i in range(len(grid)):
    for j in range(len(grid[i])):
      grid[i][j] = -1

if __name__=="__main__":
    grid = tf.convert_to_tensor(grid)
    grid = tf.expand_dims(grid,0)
    print(model(grid))
    
