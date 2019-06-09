"use strict";

let game;

class Game {
	constructor( width, height ) {
		this.width = width;
		this.height = height;
		this.bullets = new Array();
		this.aliens = Array();
	}

	initPlayerShip() {
		this.playerShip = new Ship( this.width/2, this.height, 0, 0, 0, 0, 0 );
	}

	initAliens() {
		for ( let i = 0 ; i < 300 ; i++ )
		{
			let y;
			let alien;

			y = Math.floor(Math.random()*601);

			if ( Math.floor(Math.random()*2)%2 == 0 ) {
				alien = new Ship( 0, y, Math.random(), 1, 0, 0, 1);
			} else {
				alien = new Ship( 600, y, -Math.random(), 1, 0, 0, 1);
			}

			this.aliens.unshift(alien);
		}
	}

	treatBullet( bullet ) {
		bullet.applyAcceleration();
		bullet.applySpeed();
	}

	treatBullets() {
		this.bullets.map(this.treatBullet);
	}

	displayBullet( bullet ) {
		if ( !bullet.destroy ) {
			bullet.drawBullet();
		}
	}

	displayBullets() {
		this.bullets.map(this.displayBullet);
	}

	displayAlien( alien ) {
		if ( !alien.destroy )
		{
			let canvas = document.getElementById("game_area");
			let ctx = canvas.getContext("2d");
			alien.drawShip(ctx);
		}
	}

	displayAliens() {
		this.aliens.map(this.displayAlien);
	}

	applyAlienSpeed( alien ) {
		alien.applySpeed();
	}

	applyAliensSpeed() {
		this.aliens.map(this.applyAlienSpeed);
	}

	treatBulletCollision( ship, bullet ) {
		if ( bullet.y > ship.y && bullet.y < ship.y+10 && bullet.x > ship.x-5 && bullet.x < ship.x+5 )
		{
			ship.setDestroy();
			bullet.setDestroy();
		}
	}

	treatShipCollision( ship1, ship2) {
		if ( ship1.x-5 < ship2.x+5 && ship1.x+5 > ship2.x-5 && ship1.y < ship2.y+10 && ship1.y+10 > ship2.y ) {
			ship1.setDestroy();
			ship2.setDestroy();
		}
	}

	treatCollisions() {
		for ( let j = 0 ; j < this.aliens.length ; j++ ) {
			if ( !this.aliens[j].destroy ){
				for ( let i = 0 ; i < this.bullets.length ; i++ ) {			
					if ( !this.bullets[i].destroy ) {
						this.treatBulletCollision(this.aliens[j], this.bullets[i]);
					}
				}
				this.treatShipCollision(this.playerShip,this.aliens[j]);
			}
		}
	}
}

class Ship {
	constructor( x, y, speedX, speedY, accelerationX, accelerationY, type ) {
		this.x = x;
		this.y = y;
		this.speedX = speedX;
		this.speedY = speedY;
		this.accelerationX = 0;
		this.accelerationY = 0;
		this.type = type; /* Type 0 = player, type 1 = alien*/
		this.destroy = false;
	}

	drawShip( ctx ) {
		ctx.beginPath();
		if ( this.type == 0 ) {
			ctx.strokeStyle = "#FF0000";
			ctx.moveTo(this.x, this.y);
			ctx.lineTo(this.x+5, this.y+10);
			ctx.lineTo(this.x-5, this.y+10);
			ctx.lineTo(this.x, this.y);
		} else {
			ctx.strokeStyle = "#004400";
			ctx.moveTo(this.x, this.y+10);
			ctx.lineTo(this.x+5, this.y);
			ctx.lineTo(this.x-5, this.y);
			ctx.lineTo(this.x, this.y+10);
		}
		
		ctx.stroke();
	}

	applyAcceleration() {
		this.speedX = clamp(this.speedX + this.accelerationX, -2.5, 2.5);
		this.speedY = clamp(this.speedY + this.accelerationY, -2.5, 2.5);
	}

	applySpeed() {
		if ( this.type == 0 )
		{
			this.x = clamp(this.x+this.speedX,10,590);
			this.y = clamp(this.y+this.speedY,10,590);
		} else {
			if ( clamp(this.y+this.speedY,10,590) == 10 || clamp(this.y+this.speedY,10,590) == 590 ) {
				this.speedY = -this.speedY*1.1;
			}
			this.y = clamp(this.y+this.speedY,10,590);

			if ( clamp(this.x+this.speedX,10,590) == 10 || clamp(this.x+this.speedX,10,590) == 590 ) {
				this.speedX = -this.speedX;
			}
			this.x = clamp(this.x+this.speedX,10,590);
		}
	}

	setDestroy() {
		this.destroy = true;
	}
}

class Bullet {
	constructor( x, y, speedX, speedY, accelerationX, accelerationY ) {
		this.x = x;
		this.y = y;
		this.speedX = speedX;
		this.speedY = speedY;
		this.accelerationX = accelerationX;
		this.accelerationY = accelerationY;
		this.destroy = false;
	}

	applyAcceleration() {
		this.speedX += this.accelerationX;
		this.speedY += this.accelerationY;
	}

	applySpeed() {
		this.x += this.speedX;
		this.y += this.speedY;
	}

	drawBullet() {
		/* TODO : degager ce canvas */
		let canvas = document.getElementById("game_area");
		let ctx = canvas.getContext("2d");
		ctx.beginPath();
		ctx.arc(this.x, this.y, 2, 0 , 2 * Math.PI);
		ctx.fillStyle = "white";
		ctx.fill();
	}

	setDestroy() {
		this.destroy = true;
	}
}

function clamp( a, min, max )
{
	if ( a > max ) return max;
	else if ( a < min ) return min;
	else return a;
}

window.onload = function() {
	let canvas = document.getElementById("game_area");
	let context = canvas.getContext("2d");
	context.fillStyle = "black";
	game = new Game( 600, 600 );
	game.initPlayerShip();
	game.initAliens();

	let map = {}; // You could also use an array
	onkeydown = onkeyup = function(e){
	    e = e || event; // to deal with IE
	    map[e.keyCode] = e.type == 'keydown';

	    if ( map[38] && map[40] ) /* up + down*/
	    {
	    	game.playerShip.accelerationY = 0;
	    }

	    if ( !map[38] && !map[40] ) /* ~up + ~down*/
	    {
	    	game.playerShip.accelerationY = 0;
	    }

	    if ( map[37] && map[39] ) /* left + right*/
	    {
	    	game.playerShip.accelerationX = 0;
	    }

	    if ( !map[37] && !map[39] ) /* ~left + ~right*/
	    {
	    	game.playerShip.accelerationX = 0;
	    }
	    
	    if ( map[38] ) {/* up */
	    	game.playerShip.accelerationY = clamp( game.playerShip.accelerationY-0.3, -1, 1);
	    }
	    if ( map[40] ) {/* down */
	    	game.playerShip.accelerationY = clamp( game.playerShip.accelerationY+0.3, -1, 1);;
	    }

	    if ( map[37] ) {/* left */
	    	game.playerShip.accelerationX = clamp( game.playerShip.accelerationX-0.3, -1, 1);;
	    }
	    if ( map[39] ) {/* right */
	    	game.playerShip.accelerationX = clamp( game.playerShip.accelerationX+0.3, -1, 1);;
	    }

	    if ( map[32] ) { /* space */
	    	let newBullet = new Bullet( game.playerShip.x, game.playerShip.y, 0, clamp(game.playerShip.speedY, game.playerShip.speedY, -0.5), 0, 0 );
	    	game.bullets.unshift(newBullet);
	    }
	}

	context.save();
	game.playerShip.drawShip( context );

	let interval = setInterval(() => {
		context.clearRect(0,0,600,600);
		game.playerShip.applyAcceleration();
		game.playerShip.applySpeed();
		game.playerShip.drawShip( context );
		game.applyAliensSpeed();
		game.treatBullets();
		game.treatCollisions();
		game.displayBullets();
		game.displayAliens();

		if ( game.playerShip.destroy ) clearInterval(interval);
	}, 20);
};
