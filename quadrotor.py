# -*- coding: utf-8 -*-
"""quadrotor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F0s6o9zvD0RCMyJ-th6ORmQpeo0L4mVv

Common Functions
"""

import jax
import jax.numpy as jnp
import math
import scipy.integrate
import matplotlib.pyplot as plt

# wedge operation
# x should be a column vector
def wedge(x):
    wedge_x = jnp.array([[0,-x[2][0], x[1][0]], [x[2][0], 0, -x[0][0]], [-x[1][0], x[0][0], 0]])
    return wedge_x

def split_to_states(X):
    x = X[0:3].reshape(3,1)
    v = X[3:6].reshape(3,1)
    W = X[6:9].reshape(3,1)
    R = (X[9:18]).reshape(3,3)
    return x, v, R, W

# derivative of a unit vector
def deriv_unit_vector(q, q_dot, q_ddot):
  nq = jnp.linalg.norm(q)
  u = q / nq
  u_dot = q_dot / nq - q * jnp.dot(jnp.ravel(q), jnp.ravel(q_dot)) / nq**3
  u_ddot = q_ddot / nq - q_dot / (nq**3) * (2 * jnp.dot(jnp.ravel(q), jnp.ravel(q_dot))) \
  - q / nq**3 * (jnp.dot(jnp.ravel(q_dot), jnp.ravel(q_dot)) + jnp.dot(jnp.ravel(q), jnp.ravel(q_ddot))) \
  + 3 * q / nq**5 * jnp.dot(jnp.ravel(q), jnp.ravel(q_dot))**2
  return u, u_dot, u_ddot

# vee operation
def vee(S):
  s = jnp.array([[-S[1,2]], [S[0,2]], [-S[0,1]]])
  return s

def command(t):
  desired = dict()
  desired.update({'x': jnp.array([2*(1-jnp.cos(t)), 2*jnp.sin(t), 0.1*jnp.sin(t)]).reshape(3,1)})
  desired.update({'v': jnp.array([2*jnp.sin(t), 2*jnp.cos(t), 0.1*jnp.cos(t)]).reshape(3,1)})
  desired.update({'x_2dot': jnp.array([2*(jnp.cos(t)), -2*jnp.sin(t), -0.1*jnp.sin(t)]).reshape(3,1)})
  desired.update({'x_3dot': jnp.array([-2*jnp.sin(t), -2*jnp.cos(t), -0.1*jnp.cos(t)]).reshape(3,1)})
  desired.update({'x_4dot': jnp.array([-2*jnp.cos(t), 2*jnp.sin(t), 0.1*jnp.sin(t)]).reshape(3,1)})

  desired.update({'yaw': 0})
  desired.update({'b1': jnp.array([[1], [0], [0]])})
  desired.update({'b1_dot': jnp.zeros_like(desired['b1'])})
  desired.update({'b1_2dot': jnp.zeros_like(desired['b1'])})
  return desired

# position control
def position_control(x, v, R, W, desired, k, m):
  e3 = jnp.array([[0],[0],[1]])
  g = 9.8

  error_x = x - desired['x']
  error_v = v - desired['v']
  error = {
    'x': error_x,
    'v': error_v,
    'W': 0,
    'R': 0
  }
  A = -k['x'] * error['x'] - k['v'] * error['v'] - m * g * e3 + m * desired['x_2dot']

  b3 = R @ e3
  f = -jnp.dot(jnp.ravel(A), b3)
  ev_dot = g * e3 - f / m * b3 - desired['x_2dot']
  A_dot = -k['x'] * error['v'] - k['v'] * ev_dot + m * desired['x_3dot']

  b3_dot = R @ wedge(W) @ e3
  f_dot = -jnp.dot(jnp.ravel(A_dot), b3) - jnp.dot(jnp.ravel(A), b3_dot)
  ev_2dot = -f_dot / m * b3 - f / m * b3_dot - desired['x_3dot']
  A_ddot = - k['x'] * ev_dot - k['v'] * ev_2dot + m * desired['x_4dot']

  b3c, b3c_dot, b3c_ddot = deriv_unit_vector(-A, -A_dot, -A_ddot)

  A2 = -wedge(desired['b1']) @ b3c

  A2_dot = -wedge(desired['b1_dot']) @ b3c - wedge(desired['b1']) @ b3c_dot

  A2_ddot = -wedge(desired['b1_2dot']) @ b3c - 2 * wedge(desired['b1_dot']) @ b3c_dot - wedge(desired['b1']) @ b3c_ddot

  b2c, b2c_dot, b2c_ddot = deriv_unit_vector(A2, A2_dot, A2_ddot)

  b1c = wedge(b2c) @ b3c
  b1c_dot = wedge(b2c_dot) @ b3c + wedge(b2c) @ b3c_dot
  b1c_ddot = wedge(b2c_ddot) @ b3c + 2 * wedge(b2c_dot) @ b3c_dot + wedge(b2c) @ b3c_ddot

  Rc = jnp.hstack((b1c, b2c))
  Rc = jnp.hstack((Rc, b3c))
  Rc_dot = jnp.hstack((b1c_dot, b2c_dot))
  Rc_dot = jnp.hstack((Rc_dot, b3c_dot))

  Rc_ddot = jnp.hstack((b1c_ddot, b2c_ddot))
  Rc_ddot = jnp.hstack((Rc_ddot, b3c_ddot))

  Wc = vee(jnp.transpose(Rc) @ Rc_dot)
  # print(jnp.transpose(Rc) @ Rc_ddot - wedge(Wc) @ wedge(Wc))
  Wc_dot = vee(jnp.transpose(Rc) @ Rc_ddot - wedge(Wc) @ wedge(Wc))
  return f, Rc, Wc, Wc_dot, error


# attitude control
def attitude_control(R, W, Rd, Wd, Wddot, k, J):
  eR = 1 / 2 * vee(jnp.transpose(Rd) @ R - jnp.transpose(R) @ Rd)
  eW = W - jnp.transpose(R) @ Rd @ Wd
  M = - k['R'] * eR - k['W'] * eW + wedge(W) @ (J @ W) - J @ (wedge(W) @ jnp.transpose(R) @ Rd @ Wd - jnp.transpose(R) @ Rd @ Wddot)
  return M, eR, eW

"""Dynamics"""

def Xdot(t, X, u, param):
    e3 = jnp.array([0,0,1])
    m = param['m']
    J = param['J']
    # setting values from state vector
    x, v, R, W = split_to_states(X)

    f = u[0]
    M = u[1:4]

    # evaluating dynamics
    xdot = v
    vdot = (param['g'] * e3 - (f / m * R @ e3)).reshape(3,1)
    Wdot = jnp.linalg.inv(J) @ (-wedge(W) @ J @ W + M)
    Rdot = R @ wedge(W)

    # return new state vector
    Rdot = Rdot.reshape(9,1)
    Xdot = jnp.vstack((xdot, vdot))
    Xdot = jnp.vstack((Xdot, Wdot))
    Xdot = jnp.vstack((Xdot, Rdot))
    return jnp.ravel(Xdot)

from jax.scipy.linalg import expm
def discrete_Xdot(X, u, dt, param):
  e3 = jnp.array([0,0,1])
  g = 9.8
  m = param['m']
  J = param['J']
  thrust = u[0]
  M = u[1:4]
  x, v, R, W = split_to_states(X)

  xdot = x + dt * v
  vdot = v + dt * (g * e3 - thrust / m * R @ e3).reshape(3,1)
  Wdot = W + dt * jnp.linalg.inv(J) @ (-wedge(W) @ J @ W + M)
  Rdot = (R @ expm(dt * wedge(W))).reshape(9,1)

  Xdot = jnp.vstack((xdot, vdot))
  Xdot = jnp.vstack((Xdot, Wdot))
  Xdot = jnp.vstack((Xdot, Rdot))
  return jnp.ravel(Xdot)

"""Controller"""

def geometric_controller(X, desired, k, param):
    x, v, R, W = split_to_states(X)

    f, Rc, Wc, Wc_dot, error = position_control(x, v, R, W, desired, k, param['m'])

    M, error['R'], error['W'] = attitude_control(R, W, Rc, Wc, Wc_dot, k, param['J'])

    u = jnp.vstack((f, M[0]))
    u = jnp.vstack((u, M[1]))
    u = jnp.vstack((u, M[2]))

    return u, error

def geometric_controller_for_sens(X, desired, k, param):
    k_dict = {
        'x': k[0,3],
        'v': k[3,6],
        'R': k[6,9],
        'W': k[9,12]
    }
    x, v, R, W = split_to_states(X)

    f, Rc, Wc, Wc_dot, error = position_control(x, v, R, W, desired, k_dict, param['m'])

    M, error['R'], error['W'] = attitude_control(R, W, Rc, Wc, Wc_dot, k_dict, param['J'])

    u = jnp.vstack((f, M[0]))
    u = jnp.vstack((u, M[1]))
    u = jnp.vstack((u, M[2]))

    return u

"""Sensitvity Computation"""

def sensitivityComputation(dxdtheta_current, X, u, desired, param, k):
  m, dt, J = param['m'], param['dt'], param['J']

  desired_vec = jnp.array([
      [desired['v']],
    [desired['x_2dot']],
    [desired['x_3dot']],
    [desired['x_4dot']],
    [desired['b1']],
    [desired['b1_dot']],
    [desired['b1_2dot']],
  ])

  k_vec = jnp.vstack((k['x'], k['v']))
  k_vec = jnp.vstack((k_vec, k['R']))
  k_vec = jnp.vstack((k_vec, k['W']))

  grad_f_X = jax.jacfwd(discrete_Xdot, argnums=0)
  dfdX = grad_f_X(X, u, dt, param)
  dfdX = jnp.array(dfdX)

  grad_f_u = jax.jacfwd(discrete_Xdot, argnums=1)
  dfdu = grad_f_u(X, u, dt, param)
  dfdu = jnp.array(dfdu)
  dfdu = dfdu[:,:,0]

  grad_h_X = jax.jacfwd(geometric_controller_for_sens, argnums=0)
  dhdX = grad_h_X(X, desired, k_vec, param)
  dhdX = dhdX[:,0,:]

  grad_h_theta = jax.jacfwd(geometric_controller_for_sens, argnums=2)
  dhdtheta = grad_h_theta(X, desired, k_vec, param)
  dhdtheta = dhdtheta[:,0,:,0]

  dXdphi = (dfdX + dfdu @ dhdX) @ dxdtheta_current + dfdu @ dhdtheta
  dudphi = dhdX @ dxdtheta_current + dhdtheta
  return dXdphi, dudphi

"""Run DiffTune

"""
def main():
  
  from os import XATTR_SIZE_MAX
  import jax
  import jax.numpy as jnp
  import time
  from tqdm import tqdm

  # jit compilation to save time
  jit_sensitivityComputation = jax.jit(sensitivityComputation)
  jit_Xdot = jax.jit(Xdot)
  jit_geometric_controller = jax.jit(geometric_controller)
  jit_command = jax.jit(command)
    
  param = dict()
  param.update({'dt': 0.01})
  t = jnp.arange(0, 10 + param['dt'], param['dt'])
  # t = jnp.arange(0, 2 + param['dt'], param['dt'])
  N = len(t)
  
  # moment of inertia
  J1 = 0.0820
  J2 = 0.0845
  J3 = 0.1377
  J = jnp.array([J1, J2, J3])
  param.update({'J': jnp.diag(J)})
  
  # mass
  param.update({'m': 4.34})
  param.update({'g': 9.81})
  
  # initialize controller gains
  k = {
      'x': 16*jnp.ones(3,).reshape(3,1),
      'v': 5.6*jnp.ones(3,).reshape(3,1),
      'R': 8.81*jnp.ones(3,).reshape(3,1),
      'W': 2.54*jnp.ones(3,).reshape(3,1)
  }
  
  # initialize vars for DiffTune iterations
  learningRate = 0.001
  totalIterations = 10
  itr = 0
  
  loss_history = jnp.array([])
  rmse_history = jnp.array([])
  param_history = jnp.zeros((12,1))
  gradientUpdate = jnp.zeros((12,1))
  run = True
  while run:
    itr += 1
    # load initial states
  
    p0 = jnp.zeros((3,1))
    v0 = jnp.zeros((3,1))
    R0 = jnp.eye(3)
    omega0 = jnp.array([[0], [0], [0.001]])
  
    X_storage = jnp.vstack((p0, v0))
    X_storage = jnp.vstack((X_storage, omega0))
    X_storage = jnp.vstack((X_storage, (R0.reshape(9,1))))
    desiredPosition_storage = jnp.zeros_like(v0)
  
    # initialize sensitivity (using storage)
    dx_dtheta = jnp.zeros((18,12,N+1))
    du_dtheta = jnp.zeros((4,12,N))

    # initialize loss and gradient of loss
    loss = 0
    theta_gradient = jnp.zeros((1, 12))
  
    #initialize tracking errors
    e = {
        'x': jnp.zeros((3,N)),
        'v': jnp.zeros((3,N)),
        'R': jnp.zeros((3,N)),
        'W': jnp.zeros((3,N))
    }
  
    # duration for sensitivity computation
    # duration = jnp.zeros_like(t)
  
    # for i in range(N):
    for i in tqdm(range(N), desc="DiffTune iteration"):
      #print("new time")
      X = X_storage[:,-1]
  
      # generate desired states
      desired = jit_command((i-1) * param['dt'])
      desiredPosition_storage = jnp.hstack((desiredPosition_storage, desired['x']))
  
  
      #compute control actions
      # t_ctrler = time.time()
      u, err = jit_geometric_controller(X, desired, k, param)
      # t_ctrler_f = time.time()
      # durationString = f"The geometric controller comp time is {(t_ctrler_f - t_ctrler):.4f}"
      # print(durationString)

      #sensitivity computations
      # t_seed = time.time()
      # dx_dtheta[:,:,i+1], du_dtheta[:,:,i] = sensitivityComputation(dx_dtheta[:,:,i], X, u, desired, param, k)
      new_dx_dtheta, new_du_dtheta = jit_sensitivityComputation(dx_dtheta[:, :, i], X, u, desired, param, k)
      dx_dtheta = dx_dtheta.at[:, :, i+1].set(new_dx_dtheta)
      du_dtheta = du_dtheta.at[:, :, i].set(new_du_dtheta)
      # t_seed_f = time.time()
      # duration[i] = t_seed_f - t_seed
      # duration = duration.at[i].set(t_seed_f - t_seed)

      # durationString = f"The sensitivity computation time is {(t_seed_f - t_seed):.4f}"
      # print(durationString)
  
      # store errors
      e['x'] = jnp.insert(e['x'], i, err['x'], axis=1)
      e['v'] = jnp.insert(e['v'], i, err['v'], axis=1)
      e['R'] = jnp.insert(e['R'], i, err['R'], axis=1)
      e['W'] = jnp.insert(e['W'], i, err['W'], axis=1)
  
  
      # compute error
      norm_ex = jnp.linalg.norm(err['x'])
      norm_eR = jnp.linalg.norm(err['R'])
  
      # the loss is position tracking error norm square
      loss = loss + jnp.linalg.norm(err['x'])**2
  
      # accumulating the gradient of loss wrt controller parameters
      error_matrix = jnp.vstack((err['x'], (jnp.zeros(15,)).reshape(15,1)))
      # print("dxdtheta ", dx_dtheta[:,:,i].shape)
      theta_gradient = theta_gradient + 2 * error_matrix.reshape(1,18) @ dx_dtheta[:,:,i]
      # print("theta_grad: ", theta_gradient.shape)

      # integrate the ode dynamics
      # t_ode = time.time()
      Xsol = scipy.integrate.odeint(jit_Xdot, X, [t[i], t[i+1]], tfirst=True, args = (u, param))
      # t_ode_f = time.time()
      # durationString = f"The ode integration time is {(t_ode_f - t_ode):.4f}"
      # print(durationString)

      # store the new state
      # print(Xsol)
      X_storage = jnp.hstack((X_storage, (Xsol[-1,:]).reshape(18,1)))
  
    # compute the RMSE
    RMSE = math.sqrt(1/N * loss)
    print('Iteration ', itr, 'current loss is ', loss, 'RMSE: ', RMSE)
  
    if itr >= totalIterations:
      run = False
    print(loss)
    loss_history = jnp.append(loss_history, jnp.array(loss))
    rmse_history = jnp.append(rmse_history, jnp.array(RMSE))
  
    gradientUpdate = - learningRate * theta_gradient
    # print(gradientUpdate.shape)
    # print(gradientUpdate)
    # update the parameters
    k['x'] = k['x'] + (gradientUpdate[:,0:3]).reshape(3,1)
    k['v'] = k['v'] + (gradientUpdate[:,3:6]).reshape(3,1)
    k['R'] = k['R'] + (gradientUpdate[:,6:9]).reshape(3,1)
    k['W'] = k['W'] + (gradientUpdate[:,9:12]).reshape(3,1)
  
    # projection of all parameters to be > 0.5
    threshold = 0.5
  
    if any(k['x'] < threshold):
      neg_indicator = k['x'] < threshold
      pos_indicator = ~neg_indicator
      k_default = 0.5 * jnp.ones(3)
      k['x'] = neg_indicator * k_default + pos_indicator * k['x']
  
    if any(k['v'] < threshold):
      neg_indicator = k['v'] < threshold
      pos_indicator = ~neg_indicator
      k_default = 0.5 * jnp.ones(3)
      k['v'] = neg_indicator * k_default + pos_indicator * k['v']
  
    if any(k['R'] < threshold):
      neg_indicator = k['R'] < threshold
      pos_indicator = ~neg_indicator
      k_default = 0.5 * jnp.ones(3)
      k['R'] = neg_indicator * k_default + pos_indicator * k['R']
  
    if any(k['W'] < threshold):
      neg_indicator = k['W'] < threshold
      pos_indicator = ~neg_indicator
      k_default = 0.5 * jnp.ones(3)
      k['W'] = neg_indicator * k_default + pos_indicator * k['W']
    curr_param = jnp.vstack((k['x'],k['v']))
    curr_param = jnp.vstack((curr_param,k['R']))
    curr_param = jnp.vstack((curr_param,k['W']))
    param_history = jnp.hstack((param_history,curr_param))
  
  
    # Create a single figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10))
  
    # Subplot 1: x Tracking
    ax1.plot(t, X_storage[0, 1:], label='actual', linewidth=1.5)
    ax1.plot(t, desiredPosition_storage[0, 1:], label='desired', linewidth=1.5)
    ax1.set_ylabel('x [m]')
    ax1.legend()
    ax1.grid(True)
  
    # Subplot 2: y Tracking
    ax2.plot(t, X_storage[1, 1:], label='actual', linewidth=1.5)
    ax2.plot(t, desiredPosition_storage[1, 1:], label='desired', linewidth=1.5)
    ax2.set_ylabel('y [m]')
    ax2.legend()
    ax2.grid(True)
  
    # Subplot 3: z Tracking
    ax3.plot(t, X_storage[2, 1:], label='actual', linewidth=1.5)
    ax3.plot(t, desiredPosition_storage[2, 1:], label='desired', linewidth=1.5)
    ax3.set_ylabel('z [m]')
    ax3.legend()
    ax3.grid(True)
  
    # Adjust layout and show the plot
    fig.tight_layout()
    plt.show(block=False)
  
    # ax3.set_ylabel('z [m]')
    # ax3.set_xlabel('time [s]')
    # ax3.grid(True)
  
    # # Subplot 4: RMSE Analysis
    # ax4 = plt.subplots(2)
    # ax4.plot(rmse_history, linewidth=1.5)
    # ax4.stem([len(rmse_history)], [rmse_history[-1]], linefmt='C0-', basefmt='b', markerfmt='bo')
    # ax4.set_xlim([0, 100])
    # ax4.set_ylim([0, rmse_history[0] * 1.1])
    # ax4.text(50, 0.3, f'iteration = {len(rmse_history)}', fontsize=12)
    # # ax4.set_xlabel('iterations')
    # # ax4.set_ylabel('RMSE [m]')
    # ax4.grid(True)
    # print(itr)
  
  plt.tight_layout()
  plt.show()
  
if __name__ == "__main__":
    main()