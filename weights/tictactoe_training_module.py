import tensorflow as tf
import numpy as np
from tensorflow import keras
import random

# only learns from player; n.2
# could have made it so learns from both players, but just inconvenient
# grid[i][j] -1 for none, 1 for opponent, 0 for player

def evaluate(games, episode_count):
  won = 0
  lost = 0
  _lost = 0
  tie = 0
  for n in range(games):
    done = False
    _done=False
    reset_grid()
    turn = 0
    while not done:
      if turn%2==0:
        action, _ = getAction(getState())
        result, done = step(action)
        if(done):
          if result<0:
            _lost+=1
            break
      else:
        result, done = cpu_step()
 
      if done:
        if result == 1:
          won+=1
        elif result == -1:
          lost+=1
        else:
          tie+=1
      turn+=1
  print("won:", won/games*100,"%")
  print("lost:", lost/games*100,"%")
  print("_lost: ", _lost/games*100)
  print("tie: ", tie/games*100, "%")
  if won/games*100 > 97:
    name = '{0}%%ep{1}.h5'.format(int(won/games*100*10),episode_count)
    model.save(name)

def check():
    def fullBoard():
      for i in range(len(grid)):
        for j in range(len(grid[i])):
          if grid[i][j]==-1:
            return False
      return True
    
    # Checking rows
    for i in range(3):
      if (grid[i][0] == grid[i][1] == grid[i][2]):
        if grid[i][0] == 0:
          return 1,True
        elif grid[i][0] == 1:
          return -1,True
  
      if (grid[0][i] == grid[1][i] == grid[2][i]):
        if grid[0][i] == 0:
          return 1,True
        elif grid[0][i] == 1:
          return -1,True
        
    if (grid[0][0] == grid[1][1] == grid[2][2]):
      if grid[0][0] == 0:
          return 1,True
      elif grid[0][0] == 1:
          return -1,True
      
    if (grid[0][2] == grid[1][1] == grid[2][0]):
      if grid[0][2] == 0:
          return 1,True
      elif grid[0][2] == 1:
          return -1,True
      
    if fullBoard():
      #print("full")
      return 0, True
    
    return 0,False
  
def get_model():
  '''
  inputs = keras.layers.Input(shape=(3,3,))
  pre = keras.layers.Flatten()(inputs)
  common1 = keras.layers.Dense(32,activation="relu")(pre)
  common = keras.layers.Dense(32, activation="relu")(common1)
  actor = keras.layers.Dense(9)(common)
 
  return keras.Model(inputs=inputs, outputs=[actor])
  '''
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(3,3,)),
    keras.layers.Dense(32,activation="relu"),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(9)                             
  ])
  return model

def reset_grid():
  for i in range(len(grid)):
    for j in range(len(grid[i])):
      grid[i][j] = -1
 
def step(action):
  i = action//3
  j = action - i*3
 
  if(grid[i][j]==-1):
    grid[i][j] = 0
    return check()
  else:
    return -2.5, True
 
def cpu_step():
  i = random.randrange(3)
  j = random.randrange(3)
  if grid[i][j]!=-1:
    cpu_step()
  else:
    grid[i][j] = 1
  return check()
 
def getState():
  state = tf.convert_to_tensor(grid)
  state = tf.expand_dims(state, 0)
  return state
 
def getAction(state, epsGreedy = False):
  action_value = model(state)
  action = -1
  if epsGreedy and random.random() < eps:
    while True:
      action = random.randrange(9)
      if(grid[action//3][action%3]==-1):
        return action, action_value
  else:
    action = tf.math.argmax(action_value[0])
  return action, action_value
 

# seed = 14
for seed in [14]:
  print("starting with seed {}".format(seed))
  tf.random.set_seed(seed)
  random.seed(seed)
  #eps = np.finfo(np.float32).eps.item()
  eps = 0.2

  grid = [[-1 for x in range(3)] for y in range(3)]

  model = get_model()

  optimizer = keras.optimizers.Adam(learning_rate=0.009) #0.009
  
  action_value_history = []
  rewards_history = []
  running_reward = 0
  episode_count = 0
  turn_count = 0
  reward = 0


  for i in range(17000):  # Run until solved
      reset_grid()
      episode_reward = 0
      with tf.GradientTape() as tape:
          while True:
              done = False
              _done = False
  
              state = tf.convert_to_tensor(grid)
              state = tf.expand_dims(state, 0)
  
              if turn_count%2==0:
                action, action_value = getAction(state, epsGreedy = True)
                action_value_history.append(action_value[0,action])
                reward, done = step(action)
                if done:
                  rewards_history.append(reward)
                  episode_reward = reward
                  break
              else:
                reward,done = cpu_step()
                rewards_history.append(reward)
                if done:
                  episode_reward = reward
                  break
  
              turn_count += 1
  
          # Update running reward to check condition for solving
          running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
  
          returns = []
          discounted_sum = 0
          for r in rewards_history[::-1]:
              discounted_sum = r + gamma * discounted_sum
              returns.insert(0, discounted_sum)
          
          #print("returns: ", returns)
          #print("critic_value: ", critic_value_history)
          # Calculate expected value from rewards
          # - At each timestep what was the total reward received after that timestep
          # - Rewards in the past are discounted by multiplying them with gamma
          # - These are the labels for our critic
  
          # Calculating loss values to update our network
          history = zip(action_value_history, returns)
          actor_losses = []
  
          for qhat, q in history:
              # At this point in history, the critic estimated that we would get a
              # total reward = `value` in the future. We took an action with log probability
              # of `log_prob` and ended up recieving a total reward = `ret`.
              # The actor must be updated so that it predicts an action that leads to
              # high rewards (compared to critic's estimate) with high probability.
              diff = q-qhat
              actor_losses.append(diff**2)  # actor loss
              #print(diff)
  
              # The critic must be updated so that it predicts a better estimate of
              # the future rewards.
  
          # Backpropagation
          #loss_value = sum(actor_losses)
  
          grads = tape.gradient(actor_losses, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
          # Clear the loss and reward history
          actor_losses.clear()
          action_value_history.clear()
          rewards_history.clear()
          returns.clear()
  
      # Log details
      episode_count += 1
      if episode_count % 500 == 0:
          template = "running reward: {:.2f} at episode {}"
          print(template.format(running_reward, episode_count))
          evaluate(300, episode_count)
